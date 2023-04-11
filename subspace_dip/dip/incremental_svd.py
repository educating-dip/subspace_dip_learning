import os
import scipy
import numpy as np
import tensorly as tl

class IncremetalSVD:
    def __init__(self, 
        n_eigenvecs: int,
        gamma: float = 1, 
        batch_size: int = 1
        ):
        self.n_eigenvecs = n_eigenvecs
        self.gamma = gamma
        self.batch_size = batch_size
        self.should_exclude_cnt = 0
        self.stop = False
    
    def start_tracking(self, data: np.asarray) -> None:
        self.data = data
        if self.data.shape[1] > 1: 
            rank = min(self.data.shape)
            self.U, self.s, VT = tl.partial_svd(
                    matrix=self.data, n_eigenvecs=rank)
            self.V = VT.T
        else:
            self.s = np.asarray(
                    [scipy.linalg.norm(self.data[:])]     )
            self.U = self.data / self.s
            self.V = np.asarray([[1.]])

    def _reorthogonalise(self, ) -> None:
        Q = np.zeros(self.U.shape)
        for i in range(self.U.shape[1]):
            q = self.U[:, i]
            for j in range(i):
                q = q - np.dot(q, Q[:,j])*Q[:,j]
            Q[:, i] = q / scipy.linalg.norm(q)
        self.U = Q
    
    def _should_exclude(self, K: np.ndarray ) -> bool:
        KTK = K.T @ K
        sqrt_det = scipy.linalg.det(KTK) ** .5
        eps = 2 * np.finfo(np.float32).eps
        return sqrt_det < eps

    def update(self, 
        C: np.ndarray, 
        eps: float = 1e-3, 
        use_reorthogonalise: bool = False, 
        use_should_exclude: bool = False, 
        patience: int = 1000
        ) -> None:
        
        ndim = C.ndim
        C  = C[:, None] if ndim == 1 else C
        assert (C.shape[1] < self.batch_size or C.shape[1] == self.batch_size)
        L = self.U.T @ C
        if self.batch_size > 1 :
            H = C - self.U @ L
            J, K = scipy.linalg.qr(H, mode='economic')
        else:
            UL = self.U @ L
            K = (C.T @ C - 2 * L.T @ L + UL.T @ UL) ** .5 
            J = (C - UL) / K

        if use_should_exclude: 
            if self._should_exclude(K=K):
                self.should_exclude_cnt += 1
                if self.should_exclude_cnt == patience:
                    self.stop = True
                return 
            self.should_exclude_cnt = 0

        upQ = np.concatenate([np.diag(self.s), L], axis=1)
        bttmQ = np.concatenate([np.zeros((K.shape[0], self.s.shape[0])), K], axis=1)
        Q = np.concatenate([upQ, bttmQ], axis=0)
        Up, sp, VpT = scipy.linalg.svd(Q)
        self.U = np.concatenate([self.U, J], axis=1)
        self.U = self.U @ Up
        self.U = self.U[:, :self.n_eigenvecs]
        if (np.abs(self.U[:, 0] @ self.U[:, -1]) > eps and use_reorthogonalise):
            self._reorthogonalise()
        upV = self.V @ VpT.T[:self.V.shape[1], :]
        self.V = np.concatenate([upV,  VpT.T[self.V.shape[1]:, :] ], axis=0)
        self.V = self.V[:, :self.n_eigenvecs]
        self.s = self.gamma * sp[:self.n_eigenvecs]
    
    def save_ortho_basis(self, 
        name: str = 'ortho_basis',
        ortho_basis_path: str = './'
        ):

        name = os.path.join(ortho_basis_path, name)
        os.makedirs(ortho_basis_path, exist_ok=True)
        np.savez_compressed(name, **{'ortho_basis': self.U, 'singular_values' : self.s})


