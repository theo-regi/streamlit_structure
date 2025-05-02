import numpy as np
import unittest
from products import ZCBond, FixedLeg, FloatLeg,\
    Swap, VanillaOption, OptionMarket, SSVICalibration,\
    OptionPricer, DupireLocalVol, HestonHelper, Portfolio, AutocallPricer, AutocallPricer
from utils import Rates_curve
from constants import OptionType, AutocallsType, Types
import matplotlib.pyplot as plt

class TestZCBond(unittest.TestCase):
    def setUp(self):
        """Initialize a ZCBond with a default nominal value"""
        self.bond = ZCBond()

    def test_npv_from_df(self):
        """Test NPV calculation using a discount factor (0.95) nominal is not given (=100)"""
        self.assertEqual(self.bond.get_npv_zc_from_df(0.95), 95)

    def test_npv_from_zcrate(self):
        """Test NPV calculation using rate and maturity"""
        rate, maturity = 0.05, 5
        expected_npv = 100 * (np.exp(-rate * maturity))
        self.assertEqual(self.bond.get_npv_zc_from_zcrate(rate, maturity), expected_npv)

    def test_discount_factor_from_zcrate(self):
        """Test discount factor calculation"""
        rate, maturity = 0.04, 3
        expected_df = np.exp(-rate * maturity)
        self.assertEqual(self.bond.get_discount_factor_from_zcrate(rate, maturity), expected_df)

    def test_get_zc_rate(self):
        """Test Zero-Coupon rate calculation from discount factor"""
        discount_factor, maturity = 0.90, 4
        expected_rate = -np.log(discount_factor) / maturity
        self.assertEqual(self.bond.get_zc_rate(discount_factor, maturity), expected_rate)

    def test_get_ytm(self):
        """Test Yield to Maturity calculation"""
        market_price, maturity = 80, 5
        expected_ytm = (100 / market_price) ** (1 / maturity) - 1
        self.assertEqual(self.bond.get_ytm(market_price, maturity), expected_ytm)

    def test_get_duration_macaulay(self):
        """Test Macaulay duration (should return maturity)"""
        maturity = 7
        self.assertEqual(self.bond.get_duration_macaulay(maturity), maturity)

    def test_get_modified_duration(self):
        """Test Modified Duration calculation"""
        market_price, maturity = 85, 4
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_mod_duration = maturity / (1 + ytm)
        self.assertEqual(self.bond.get_modified_duration(market_price, maturity), expected_mod_duration)

    def test_get_sensitivity(self):
        """Test Sensitivity calculation"""
        new_rate, maturity = 0.03, 6
        expected_sensitivity = maturity / (1 + new_rate)
        self.assertEqual(self.bond.get_sensitivity(new_rate, maturity), expected_sensitivity)

    def test_get_convexity_issued(self):
        """Test Convexity of an issued Zero-Coupon Bond"""
        maturity, market_price = 5, 80
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_convexity = (maturity * (maturity + 1) * 100) / (market_price * ((1 + ytm) ** (maturity + 2)))
        self.assertEqual(self.bond.get_convexity(maturity, market_price=market_price), expected_convexity)

    def test_get_convexity_not_issued(self):
        """Test Convexity of a non-issued Zero-Coupon Bond"""
        maturity, discount_factor = 5, 0.85
        market_price = 85
        ytm = (100 / market_price) ** (1 / maturity) - 1
        expected_convexity = (maturity * (maturity + 1) * 100) / (market_price * ((1 + ytm) ** (maturity + 2)))
        self.assertEqual(self.bond.get_convexity(maturity, discount_factor=discount_factor), expected_convexity)

    def test_invalid_convexity_inputs(self):
        """Test error handling for incorrect convexity inputs"""
        maturity = 5
        with self.assertRaises(ValueError):
            self.bond.get_convexity(maturity)

    def test_invalid_ytm_maturity(self):
        """Test YTM with zero maturity should raise an error"""
        with self.assertRaises(ValueError):
            self.bond.get_ytm(90, 0)

class TestFixedLeg(unittest.TestCase):
    def setUp(self):
        """Initialize a FixedLeg with a default nominal value"""
        rate_curve = Rates_curve("data\RateCurve.csv", 5)
        discount_curve = Rates_curve("data\RateCurve.csv")
        self.fixed_leg = FixedLeg(rate_curve, "07/03/2025", "07/03/2030", "annually", "EUR", "30/360", "Modified Following", discount_curve, 100, 0, "%d/%m/%Y", "Nelson_Siegel", True)

    def test_npv(self):
        """Test NPV calculation using a flat curve and different discount curve, nominal = 100"""
        target = 111.3299
        print(self.fixed_leg._rates_c)
        self.assertAlmostEqual(self.fixed_leg.calculate_npv(self.fixed_leg._cashflows), target, places=3)

    def test_calculate_duration(self):
        """Test Duration calculation"""
        target_duration = 4.569204416
        self.assertAlmostEqual(self.fixed_leg.calculate_duration(), target_duration, places=6)

    def test_calculate_sensitivity(self):
        """Test Sensitivity calculation: +1bp change in markets rates"""
        target_sensitivity = -0.05085670
        self.assertAlmostEqual(self.fixed_leg.calculate_sensitivity(), target_sensitivity, places=6)

    def test_calculate_pv01(self):
        """Test PV01 calculation"""
        target_pv01 = 0.046687947
        self.assertAlmostEqual(self.fixed_leg.calculate_pv01(), target_pv01, places=6)

    def test_calculate_convexity(self):
        """Test Convexity calculation"""
        target_convexity = 1.319688658
        self.assertAlmostEqual(self.fixed_leg.calculate_convexity(), target_convexity, places=6)

    def test_yield(self):
        """Test Yield to Maturity calculation"""
        target_ytm = 3.880628
        self.assertAlmostEqual(self.fixed_leg.calculate_yield(105), target_ytm, places=6)

class TestFloatLeg(unittest.TestCase):
    def setUp(self):
        """Initialize a FloatLeg with a default nominal value"""
        rate_curve = Rates_curve("data\RateCurve.csv")
        discount_curve = Rates_curve("data\RateCurve.csv")
        self.float_leg = FloatLeg(rate_curve, "07/03/2025", "07/03/2030", "annually", "EUR", "30/360", "Modified Following", discount_curve, 100, 0, "%d/%m/%Y", "Nelson_Siegel", False)


    def test_npv(self):
        """Test NPV calculation using a flat curve and different discount curve, nominal = 100"""
        target = 11.90599
        self.assertAlmostEqual(self.float_leg.calculate_npv(self.float_leg._cashflows), target, places=3)

    def test_calculate_duration(self):
        """Test Duration calculation"""
        target_duration = 3.288075
        self.assertAlmostEqual(self.float_leg.calculate_duration(), target_duration, places=6)

    def test_calculate_sensitivity(self):
        """Test Sensitivity calculation"""
        target_sensitivity = 0.03225804
        self.assertAlmostEqual(self.float_leg.calculate_sensitivity(), target_sensitivity, places=6)

    def test_calculate_pv01(self):
        """Test PV01 calculation"""
        target_pv01 = 0.046688
        self.assertAlmostEqual(self.float_leg.calculate_pv01(), target_pv01, places=6)

    def test_calculate_convexity(self):
        """Test Convexity calculation"""
        target_convexity = 0.4000001
        self.assertAlmostEqual(self.float_leg.calculate_convexity(), target_convexity, places=6)

    def test_yield(self):
        """Test Yield to Maturity calculation"""
        target_ytm = 4.880054
        self.assertAlmostEqual(self.float_leg.calculate_yield(11), target_ytm, places=4)

    def test_cap_npv(self):
        """Test Cap NPV calculation"""
        target_cap_npv = 1.504774
        self.float_leg.cap_value(0.025,0.06)
        dict_npv = self.float_leg._cashflows_cap
        result = self.float_leg.calculate_npv(dict_npv)
        self.assertAlmostEqual(result, target_cap_npv, places=6)

    def test_floor_npv(self):
        """Test Cap NPV calculation"""
        target_floor_npv = 1.270764
        self.float_leg.floor_value(0.025,0.06)
        dict_npv = self.float_leg._cashflows_floor
        result = self.float_leg.calculate_npv(dict_npv)
        self.assertAlmostEqual(result, target_floor_npv, places=6)

class TestSwap(unittest.TestCase):
    def setUp(self):
        """Initialize a FloatLeg with a default nominal value"""
        rate_curve = Rates_curve("data\RateCurve.csv")
        discount_curve = Rates_curve("data\RateCurve.csv")
        self.swap = Swap(rate_curve, "07/03/2025", "07/03/2030", "annually", "EUR", "30/360", "Modified Following", discount_curve, 100, 0, "%d/%m/%Y", "Nelson_Siegel", False)

    def test_calculate_fixed_rate(self):
        """Test Fixed Rate calculation"""
        target_fixed_rate = 0.025501
        fixed_leg_rate = self.swap.calculate_fixed_rate()

        print(self.swap.fixed_leg._rates_c)
        print(self.swap.float_leg._rates_c)
        self.assertAlmostEqual(fixed_leg_rate, target_fixed_rate, places=6)

    def test_collar(self):
        """Test Collar calculation"""
        target_collar = -0.156183
        result= self.swap.calculate_collar(0.035,0.02,0.06)
        self.assertAlmostEqual(result, target_collar, places=6)

class TestVanillaOption(unittest.TestCase):
    def setUp(self):
        self.spot=100 #Quite useless because base_spot = 100
    
    def test_payoff_call(self):
        self.vanilla_option = VanillaOption("28/03/2025", "28/03/2026", OptionType.CALL, strike=95, notional=1)
        """Test Payoff calculation"""
        self.assertEqual(self.vanilla_option.payoff(self.spot), 5)

    def test_npv_call(self):
        self.vanilla_option = VanillaOption("28/03/2025", "28/03/2026", OptionType.CALL, strike=95, notional=1)
        """Test NPV calculation using a 0.05 Interest rate"""
        self.assertAlmostEqual(self.vanilla_option.npv(self.spot), 4.7561, places=4)

    def test_pay_put(self):
        self.vanilla_option = VanillaOption("28/03/2025", "28/03/2026", OptionType.PUT, strike=100, notional=1)
        """Test Payoff calculation"""
        self.assertEqual(self.vanilla_option.payoff(self.spot), 0)

    def test_npv_put(self):
        self.vanilla_option = VanillaOption("28/03/2025", "28/03/2026", OptionType.PUT, strike=100, notional=1)
        """Test NPV calculation using a 0.05 Interest rate"""
        self.assertAlmostEqual(self.vanilla_option.npv(self.spot), 0, places=4)

class TestOptionMarket(unittest.TestCase):
    def setUp(self):
        """Set up the Option market object helper"""
        self.option_market = OptionMarket("data/options.csv", "data/underlying_prices.csv")

    def test_split_price_dates(self):
        dfs = self.option_market._dict_df
        self.assertEqual(len(dfs.keys()), 3)

    def test_build_options_matrix(self):
        matrices = self.option_market._options_matrices
        print(matrices[list(matrices.keys())[0]])
        self.assertEqual(len(matrices.keys()), 3)

    def test_get_options_for_moneyness(self):
        """Test the get_options_for_moneyness method"""
        spot = 209.68
        OTM_v = True
        p_date = "13/03/2025"
        maturity = "16/05/2025" #Checking first pricing date, first maturity
        options = self.option_market.get_options_for_moneyness(p_date, maturity, (0.7,1.3), OTM_v)
        self.assertIsInstance(options, list)
               
        moneyness_l_c, moneyness_l_p = [], []
        for option in options:
            if option._type == OptionType.CALL:
                moneyness_l_c.append(np.log(option._strike/spot))
            elif option._type == OptionType.PUT:
                moneyness_l_p.append(np.log(option._strike/spot))
        moneyness_l = moneyness_l_p + moneyness_l_c

        self.assertTrue(all(np.log(0.7) <= m <= np.log(1.3) for m in moneyness_l))
        if OTM_v == True:
            self.assertTrue(all(m >= 0 for m in moneyness_l_c))
            self.assertTrue(all(m <= 0 for m in moneyness_l_p))

class TestSSVI(unittest.TestCase):
    def setUp(self):
        """Set up the SSVI object"""
        self.ssvi = SSVICalibration("Black-Scholes-Merton", "data/options.csv", "data/underlying_prices.csv", "12/03/2025")

    def test_calibrate(self):
        """Test calibration of SSVI parameters"""
        params = self.ssvi.get_ssvi_params()
        print(params)
        self.assertEqual(isinstance(params, dict), True)
        self.assertEqual(len(list(params.values())), 6)

class TestLocalVolMatrix(unittest.TestCase):
    def setUp(self):
        """Set up the localvol object"""
        self.local_vol = DupireLocalVol("Black-Scholes-Merton", "data/options.csv", "data/underlying_prices.csv", "13/03/2025")
        
    def test_get_base_vol_matrix(self):
        import pandas as pd
        vol_matrix = self.local_vol.get_implied_vol_matrix()
        #print(vol_matrix)
        #vol_matrix.to_excel("data/vol_matrix.xlsx")
        self.assertEqual(isinstance(vol_matrix, pd.DataFrame),True)

    def test_get_local_implied_vol(self):
        local_vol = self.local_vol.get_local_implied_vol(0.5, 190)
        self.assertEqual(isinstance(local_vol, float),True)
        self.assertAlmostEqual(local_vol, 0.32776, places=4)

class TestPricer(unittest.TestCase):
    def setUp(self):
        self.pricer=OptionPricer(
            start_date="28/03/2025",
            end_date="28/03/2026",
            type=OptionType.CALL,
            model="Black-Scholes-Merton",
            spot=100,
            strike=105,
            div_rate=0.03,
            currency="EUR",
            sigma=0.08,
            rate=0.05,
            notional=1)
        """
        self.pricer=OptionPricer(
            start_date="13/03/2025",
            end_date="16/05/2025",
            type=OptionType.CALL,
            model="Black-Scholes-Merton",
            spot=209.68,
            strike=210,
            currency="USD",
            sigma=0.33,
            notional=1)"""

    def test_price(self):
        # Example test case for price calculation
        self.assertAlmostEqual(self.pricer.price, 1.925485846, places=4)

    def test_delta(self):
        # Example test case for delta calculation
        self.assertAlmostEqual(self.pricer.delta, 0.36346171, places=4)

    def test_gamma(self):
        # Example test case for gamma calculation
        self.assertAlmostEqual(self.pricer.gamma, 0.045980369, places=4)

    def test_vega(self):
        # Example test case for vega calculation
        self.assertAlmostEqual(self.pricer.vega, 36.78429503, places=4)

    def test_theta(self):
        # Example test case for theta calculation
        self.assertAlmostEqual(self.pricer.theta, -2.465482639, places=6)

    def test_rho(self):
        # Example test case for rho calculation
        self.assertAlmostEqual(self.pricer.rho, 34.42068517, places=4)

    def test_payoff(self):
        # Example test case for payoff calculation
        self.pricer.price
        self.assertAlmostEqual(self.pricer._payoff, 2.024207616, places=4)

class TestHestonCalibration(unittest.TestCase):    
    def test_heston_calibration_SSVI(self):
        self._heston_calibrator=HestonHelper("Black-Scholes-Merton", "data/options.csv", "data/underlying_prices.csv", "13/03/2025", type_calibration='SSVI')
        params = self._heston_calibrator.get_heston_params()
        print(params)
        self.assertEqual(isinstance(params, dict), True)

    def test_heston_calibration_Market(self):
        self._heston_calibrator=HestonHelper("Black-Scholes-Merton", "data/options.csv", "data/underlying_prices.csv", "13/03/2025", type_calibration='Market')
        params = self._heston_calibrator.get_heston_params()
        print(params)
        self.assertEqual(isinstance(params, dict), True)

class TestHestonPricing(unittest.TestCase):
    def setUp(self):
            params = {'v0': np.float64(0.2426087693130581), 'kappa': np.float64(0.10072759576180132), 'theta': np.float64(0.10072759576180132), 'eta': np.float64(0.1), 'rho': np.float64(-0.012369127944111824)}

            self.pricer=OptionPricer(
                start_date="13/03/2025",
                end_date="16/05/2025",
                type=OptionType.CALL,
                model="Heston",
                spot=209.68,
                strike=210,
                currency="USD",
                notional=1,
                model_parameters=params,
                nb_paths=1000,
                nb_steps=1000)
            
            """
            self.pricer=OptionPricer(
                start_date="13/03/2025",
                end_date="16/05/2025",
                type=BarrierType.CALL_UP_OUT,
                model="Heston",
                spot=209.68,
                barrier_strike=240,
                strike=210,
                currency="USD",
                notional=1,
                model_parameters=params,
                nb_paths=1000,
                nb_steps=1000)"""

    def test_price_payoffs(self):
        price = self.pricer.price
        payoffs, spots =self.pricer._payoff, self.pricer._spots_paths
        plt.scatter(spots, payoffs, label='Call Payoff', s=10)
        plt.show()
        self.assertAlmostEqual(price, 12.17, places=1)
        self.assertEqual(isinstance(payoffs, list), True)

    def test_delta(self):
        delta = self.pricer.delta
        print(delta)
        self.assertAlmostEqual(delta, 0.6, places=1)

    def test_gamma(self):
        gamma = self.pricer.gamma
        self.assertAlmostEqual(gamma, 0.0081, places=1)

    def test_vega(self):
        vega = self.pricer.vega
        self.assertAlmostEqual(vega, 3.5/100, places=1)

    def test_theta(self):
        theta = self.pricer.theta
        self.assertAlmostEqual(theta, -54.07, places=1)
    
    def test_rho(self):
        rho = self.pricer.rho
        self.assertAlmostEqual(rho, 17.17/100, places=1)

class TestDupireDiffusion(unittest.TestCase):
    def setUp(self):
        self.pricer=OptionPricer(
            start_date="13/03/2025",
            end_date="16/05/2025",
            type=OptionType.CALL,
            model="Dupire",
            spot=209.68,
            strike=210,
            currency="USD",
            notional=1,
            data_path="data/options.csv",
            file_name_underlying="data/underlying_prices.csv",
            nb_paths=1000,
            nb_steps=1000)

    def test_price(self):
        price = self.pricer.price
        payoffs, spots =self.pricer._payoff, self.pricer._spots_paths
        plt.scatter(spots, payoffs, label='Call Payoff', s=10)
        plt.show()
        self.assertAlmostEqual(price, 12.17, places=1)
        self.assertEqual(isinstance(payoffs, list), True)

    def test_delta(self):
        delta = self.pricer.delta
        self.assertAlmostEqual(delta, 0.6, places=1)
    
    def test_gamma(self):
        gamma = self.pricer.gamma
        self.assertAlmostEqual(gamma, 0.0081, places=1)

    def test_vega(self):
        vega = self.pricer.vega
        print(vega)
        self.assertAlmostEqual(vega, 3.4403, places=1)

    def test_theta(self):
        theta = self.pricer.theta
        self.assertAlmostEqual(theta, -61.85, places=1)
    
    def test_rho(self):
        rho = self.pricer.rho
        self.assertAlmostEqual(rho, 18.15, places=1)
    
class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.ptf = Portfolio()
        params = {'v0': np.float64(0.2426087693130581), 'kappa': np.float64(0.10072759576180132), 'theta': np.float64(0.10072759576180132), 'eta': np.float64(0.1), 'rho': np.float64(-0.012369127944111824)}
        self.ptf._add_product("Call", "13/03/2025", "16/05/2025", 1, 0.1, None, "Dupire", notional=1)
        self.ptf._add_product("Put Down and Out", "13/03/2025", "16/05/2025", 1, 210, 180, "Dupire", notional=1)
        self.ptf._add_product("Call Up and Out", "13/03/2025", "16/05/2025", 1, 210, 240, "Dupire", notional=1)
        self.ptf._add_product("Put", "13/03/2025", "16/05/2025", 1, 210, 180, "Heston", notional=1, heston_parameters=params)

    def test_price_portfolio(self):
        price, npvs, payoffs, spots = self.ptf.price_portfolio()
        plt.scatter(spots, payoffs, label='Payoff', s=10)
        plt.show()
        self.assertAlmostEqual(price, 12.17, places=1)
        self.assertEqual(isinstance(payoffs, list), True)

class TestAutocalls(unittest.TestCase):
    """
    Class to test both AutocallPricer and Autocalls class:
    """
    
    def setUp(self):
        params = params = {'v0': np.float64(0.2426087693130581), 'kappa': np.float64(0.10072759576180132), 'theta': np.float64(0.10072759576180132), 'eta': np.float64(0.1), 'rho': np.float64(-0.012369127944111824)}
        
        self._pricer = AutocallPricer(
            start_date="13/03/2025",
            end_date="16/02/2027",
            type=AutocallsType.PHOENIX,
            model="Heston",
            spot=209.68,
            strike=1.2,
            coupon=0.1,
            coupon_strike=1.0,
            protection=0.8,
            memory=True,
            exercise_type=Types.EUROPEAN,
            frequency="monthly",
            currency="USD",
            notional=1,
            model_parameters=params,
            nb_paths=1000,
            nb_steps=1000)
        """
        self._pricer = AutocallPricer(
            start_date="13/03/2025",
            end_date="16/02/2027",
            type=AutocallsType.AUTOCALL,
            model="Heston",
            spot=209.68,
            strike=1.5,
            final_strike=1.5,
            coupon_strike=0.9,
            coupon=0.05,
            protection=0.5,
            memory=False,
            exercise_type=Types.EUROPEAN,
            frequency="quaterly",
            currency="USD",
            notional=1,
            model_parameters=params,
            nb_paths=10000,
            nb_steps=10000)
        """
        pass

    def test_price(self):
        npv, payoff, par_cpn, call_prob = self._pricer.price
        #print(payoff)
        print(npv)
        print(par_cpn)

        import pandas as pd
        call_prob=pd.DataFrame(call_prob, columns=["YearFraction", "CumulativeProbability"])
        print(call_prob)
        plt.scatter(call_prob['YearFraction'], call_prob['CumulativeProbability'], label='Call Probability', s=10)
        plt.xlabel("Time")
        plt.show()
        pass

if __name__ == "__main__":
    unittest.main()