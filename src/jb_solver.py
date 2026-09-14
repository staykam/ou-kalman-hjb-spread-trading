import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import njit

@njit
def psor_solver(A_data, A_indices, A_indptr, b, obstacle, V_guess, omega=1.2, tol=1e-6, max_iter=1000):
    """
    Projected Successive Over-Relaxation (PSOR) to solve the Linear Complementarity Problem (LCP).
    This physically finds the "Free Boundary" (the optimal execution threshold).
    """
    n = len(b)
    V = V_guess.copy()
    
    for iteration in range(max_iter):
        error = 0.0
        for i in range(n):
            # Extract row i from CSR matrix
            row_start = A_indptr[i]
            row_end = A_indptr[i+1]
            
            sigma_sum = 0.0
            diag = 1.0
            for j_idx in range(row_start, row_end):
                j = A_indices[j_idx]
                val = A_data[j_idx]
                if i == j:
                    diag = val
                else:
                    sigma_sum += val * V[j]
                    
            # Gauss-Seidel step
            v_new = (b[i] - sigma_sum) / diag
            
            # Successive Over-Relaxation
            v_sor = V[i] + omega * (v_new - V[i])
            
            # The Projection (The "American Option" Early Exercise constraint)
            v_proj = max(v_sor, obstacle[i])
            
            error += abs(v_proj - V[i])
            V[i] = v_proj
            
        if error < tol:
            break
            
    return V

class HJBOptimalStopping:
    def __init__(self, z_min=-5.0, z_max=5.0, grid_points=1000):
        self.z_min = z_min
        self.z_max = z_max
        self.N = grid_points
        self.dz = (z_max - z_min) / (grid_points - 1)
        self.Z_grid = np.linspace(z_min, z_max, grid_points)

    def solve_entry_boundary(self, mu, theta, sigma, friction, discount_rate=0.05):
        """
        Builds the massive Finite Difference tridiagonal matrix for the OU process
        and solves the HJB Variational Inequality to find the optimal entry spread.
        """
        # 1. Construct the Infinitesimal Generator Matrix (L) for the OU process
        # L(V) = 0.5 * sigma^2 * V'' + theta * (mu - Z) * V'
        
        diag = np.zeros(self.N)
        lower = np.zeros(self.N - 1)
        upper = np.zeros(self.N - 1)
        
        sig2 = sigma**2
        dz2 = self.dz**2
        
        for i in range(1, self.N - 1):
            z = self.Z_grid[i]
            drift = theta * (mu - z)
            
            # Central difference for V' and V''
            lower[i-1] = (sig2 / (2 * dz2)) - (drift / (2 * self.dz))
            diag[i]   = -(sig2 / dz2) - discount_rate
            upper[i]   = (sig2 / (2 * dz2)) + (drift / (2 * self.dz))
            
        # Dirichlet boundary conditions (Value is zero at extreme infinities if we don't trade)
        diag[0] = 1.0
        diag[-1] = 1.0
        upper[0] = 0.0
        lower[-1] = 0.0

        A = sparse.diags([lower, diag, upper], offsets=[-1, 0, 1], format='csr')
        
        # 2. Define the Obstacle (The physical payoff of crossing the spread)
        # If we go LONG: We expect the spread to revert from Z_grid to mu, minus friction
        long_payoff = (mu - self.Z_grid) - friction
        
        # 3. Solve the Linear Complementarity Problem via PSOR
        b = np.zeros(self.N)
        V_guess = np.maximum(long_payoff, 0)
        
        V_optimal = psor_solver(A.data, A.indices, A.indptr, b, long_payoff, V_guess)
        
        # 4. Locate the Free Boundary
        # The optimal execution threshold is the exact coordinate where the Value Function 
        # is physically tangent to the Payoff function.
        exercise_region = np.isclose(V_optimal, long_payoff, atol=1e-4)
        
        # The boundary is the point closest to the mean where exercise is optimal
        try:
            boundary_idx = np.where(exercise_region & (self.Z_grid < mu))[0][-1]
            optimal_z_long = self.Z_grid[boundary_idx]
        except IndexError:
            optimal_z_long = -np.inf # Friction is too high; mathematically impossible to profit
            
        return optimal_z_long

# Example Usage:
#solver = HJBOptimalStopping()
# optimal_entry = solver.solve_entry_boundary(mu=0.0, theta=4.0, sigma=0.02, friction=0.005)