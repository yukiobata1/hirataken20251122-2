
import unittest
import numpy as np

class TestEPSROptimization(unittest.TestCase):
    def setUp(self):
        # Constants from the project
        self.c_Ga = 0.858
        self.c_In = 0.142
        self.b_Ga = 7.288
        self.b_In = 4.061
        
        # Calculate expected weights
        self.w_GaGa = (self.c_Ga * self.b_Ga)**2
        self.w_InIn = (self.c_In * self.b_In)**2
        self.w_GaIn = 2 * (self.c_Ga * self.b_Ga) * (self.c_In * self.b_In)
        
        self.w_max = max(self.w_GaGa, self.w_InIn, self.w_GaIn)
        
        # Normalized weights
        self.norm_GaGa = self.w_GaGa / self.w_max
        self.norm_InIn = self.w_InIn / self.w_max
        self.norm_GaIn = self.w_GaIn / self.w_max

    def test_weight_calculation(self):
        """Verify the physics of the weighting factors"""
        print(f"\nCalculated Weights:")
        print(f"Ga-Ga: {self.w_GaGa:.2f} (Norm: {self.norm_GaGa:.4f})")
        print(f"In-In: {self.w_InIn:.2f} (Norm: {self.norm_InIn:.4f})")
        print(f"Ga-In: {self.w_GaIn:.2f} (Norm: {self.norm_GaIn:.4f})")
        
        # Assertions
        # Ga-Ga should be dominant
        self.assertTrue(self.w_GaGa > self.w_InIn * 10, "Ga-Ga should be much stronger than In-In")
        self.assertTrue(self.w_GaGa > self.w_GaIn, "Ga-Ga should be stronger than Ga-In")
        
        # In-In should be very weak (< 1% of max is typical for this mixture)
        self.assertLess(self.norm_InIn, 0.05, "In-In contribution should be small")

    def test_convergence_toy_model(self):
        """Simulate the optimization loop with and without weights"""
        
        # Toy model: g_obs = wA*gA + wB*gB
        # We want to find U_A, U_B. 
        # Assume linear response: g_i = -U_i
        
        # Target values
        U_A_target = 1.0
        U_B_target = -0.5
        
        # Weights
        wA = self.norm_GaGa  # ~1.0
        wB = self.norm_InIn  # ~0.008
        
        # Observed signal
        g_obs = wA * (-U_A_target) + wB * (-U_B_target)
        
        # Optimization
        U_A = 0.0
        U_B = 0.0
        alpha = 0.5
        
        print("\nOptimization Trace (Weighted):")
        for i in range(20):
            g_sim = wA * (-U_A) + wB * (-U_B)
            diff = g_sim - g_obs
            
            # Weighted Update (The Fix)
            # U_new = U_old + alpha * weight * diff
            # (Note: Sign depends on definition, here assumes diff = g_sim - g_obs)
            # If g_sim > g_obs (too much structure), we increase U (repulsion) -> reduces g
            
            U_A += alpha * wA * diff
            U_B += alpha * wB * diff
            
            if i % 5 == 0:
                print(f"Iter {i}: UA={U_A:.3f}, UB={U_B:.3f}, Error={diff:.6f}")
                
        # Check convergence
        # With weighted updates, UA should get close to target, UB should stay close to 0
        # (Because we can't see B, we shouldn't move it)
        
        print(f"Final: UA={U_A:.3f}, UB={U_B:.3f}")
        
        # UA should be reasonably close (it carries the signal)
        self.assertAlmostEqual(U_A, U_A_target, delta=0.2)
        
        # UB should NOT match UA (which was the bug). It should stay small.
        self.assertNotAlmostEqual(U_B, U_A, delta=0.1)
        self.assertTrue(abs(U_B) < 0.2, "Invisible component should not move much")

if __name__ == '__main__':
    unittest.main()
