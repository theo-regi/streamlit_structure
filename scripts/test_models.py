import unittest
from models import BSM
from products import VanillaOption
from constants import OptionType

class TestBSMModel(unittest.TestCase):
    def setUp(self):
        """
        Set up of Vanilla Options for testing BSM
        """
        self.option_call = VanillaOption(
            start_date="28/03/2025", end_date="28/03/2026", type=OptionType.CALL, strike=105, rate=0.05, div_rate=0.03)
        
        self.option_put = VanillaOption(
            start_date="28/03/2025", end_date="28/03/2026", type=OptionType.PUT, strike=95, rate=0.05, div_rate=0.03)
        
        self.sigma = 0.08
        self.bsm_call = BSM(self.option_call, self.sigma)
        self.bsm_put = BSM(self.option_put, self.sigma)
        self.spot = 100

    def test_price_call(self):
        """
        Test the price calculation for a call option.
        """
        price = self.bsm_call.price(self.spot)
        self.assertAlmostEqual(price, 1.92586, places=3)

    def test_price_put(self):
        """
        Test the price calculation for a put option.
        """
        price = self.bsm_put.price(self.spot)
        self.assertAlmostEqual(price, 0.76428469, places=3)

    def test_delta_call(self):
        """
        Test the delta calculation for a call option.
        """
        delta = self.bsm_call.delta(self.spot)
        print("delta",delta)
        self.assertAlmostEqual(delta, 0.36346171, places=4)

    def test_delta_put(self):
        """
        Test the delta calculation for a put option.
        """
        delta = self.bsm_put.delta(self.spot)
        self.assertAlmostEqual(delta, -0.170685653, places=4)

    def test_gamma(self):
        """
        Test the gamma calculation for both call and put options.
        """
        gamma_call = self.bsm_call.gamma(self.spot)
        gamma_put = self.bsm_put.gamma(self.spot)
        self.assertAlmostEqual(gamma_call, 0.045980369, places=4)
        self.assertAlmostEqual(gamma_put, 0.031369622, places=4)

    def test_vega(self):
        """
        Test the vega calculation for both call and put options.
        """
        vega_call = self.bsm_call.vega(self.spot)
        vega_put = self.bsm_put.vega(self.spot)
        self.assertAlmostEqual(vega_call, 36.78429503, places=4)
        self.assertAlmostEqual(vega_put, 25.09569771, places=4)

    def test_theta_call(self):
        """
        Test the theta calculation for a call option.
        """
        theta = self.bsm_call.theta(self.spot)
        self.assertAlmostEqual(theta, -2.465482639, places=3)

    def test_theta_put(self):
        """
        Test the theta calculation for a put option.
        """
        theta = self.bsm_put.theta(self.spot)
        self.assertAlmostEqual(theta, -1.711705172, places=3)

    def test_rho_call(self):
        """
        Test the rho calculation for a call option.
        """
        rho = self.bsm_call.rho(self.spot)
        self.assertAlmostEqual(rho, 34.42068517, places=3)

    def test_rho_put(self):
        """
        Test the rho calculation for a put option.
        """
        rho = self.bsm_put.rho(self.spot)
        self.assertAlmostEqual(rho, -17.83284996, places=3)

if __name__ == '__main__':
    unittest.main()