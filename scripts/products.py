#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes de produits----------------------------
#-------------------------------------------------------------------------------------------------------
from constants import OptionType, BarrierType, AutocallsType, Types, BASE_STRIKE, BASE_RATE, BASE_CURRENCY, \
    BASE_DIV_RATE, BOUNDS_MONEYNESS, OTM_CALIBRATION, VOLUME_CALIBRATION, VOLUME_THRESHOLD, \
    INITIAL_SSVI, SSVI_METHOD, OPTIONS_SOLVER_SSVI,BASE_NOTIONAL, CONVENTION_DAY_COUNT, \
    ROLLING_CONVENTION, FORMAT_DATE, TYPE_INTERPOL, EXCHANGE_NOTIONAL, BASE_SHIFT,BASE_MODEL,\
    BASE_SIGMA, BASE_METHOD_VOL, TOLERANCE, MAX_ITER, BOUNDS, STARTING_POINT, BASE_DELTA_K, BASE_LIMITS_K, \
    BASE_CALIBRATION_HESTON, BASE_MAX_T, BASE_T_INTERVAL, INITIAL_HESTON, HESTON_BOUNDS,\
    HESTON_CALIBRATION_OPTIONS, HESTON_METHOD, BASE_LIMITS_K_H, CUTOFF_H, N_CORES, NUMBER_PATHS_H,\
    NB_STEPS_H, FILE_PATH, FILE_UNDERLYING, BASE_MODEL_AUTOCALLS, SOLVER_METHOD, DICT_PRODUCT, get_from_cache, set_in_cache

from joblib import Parallel, delayed
from functools import lru_cache
from scipy.optimize import minimize
from scipy.integrate import quad
import numpy as np
from abc import ABC, abstractmethod
from scripts.utils import PaymentScheduleHandler, Rates_curve, ImpliedVolatilityFinder, SVIParamsFinder, get_heston_params_from_csv
from scripts.models import BSM, Heston, Dupire
import scripts.utils as utils
import pandas as pd
from scipy.stats import norm
from collections import defaultdict

#Supported models for options pricing
dict_models = {"Black-Scholes-Merton": BSM, "Heston": Heston, "Dupire": Dupire}

#____________________________Classe pour les ZC (pas de call de l'abstraite)_____________________________
#Zero-Coupon Class:: nominal (optional = 100 if not given)
class ZCBond():
    """
    Classe qui cherche à répliquer un bond zero coupon = 1 paiement unique à maturité.
    
    Input: Nominal (optional)

    - NPV par un discount factor:: discount factor
    - NPV par un taux et une maturité:: rate, maturity
    - Récupérer le taux correspondant:: discount factor, maturity

    Nous implémentons que les fonctions/données de bases, on pourrait imaginer le risque de liquidité/crédit
    """
    def __init__(self, nominal: float=BASE_NOTIONAL) -> None:
        self.__nominal = nominal
        pass

    def get_npv_zc_from_df(self, discount_factor: float) -> float:
        """
        Input: discount factor (ex: 0.98) as float
        
        Returns: Net Present Value of the Zero Coupon Bond.
        """
        return self.__nominal * discount_factor

    def get_npv_zc_from_zcrate(self, rate: float, maturity: float) -> float:
        """
        Calculate Net Present Value from Zero Coupon Rate
        
        Input: Zero-coupon rate, maturity of the Zero Coupon bond
        """
        return  np.exp(-rate * maturity) * self.__nominal

    def get_discount_factor_from_zcrate(self, rate: float, maturity: float) -> float:
        """
        Calculate the discount factor from the zero-coupon rate and maturity.
        
        Input: Zero-counpon rate, maturity
        """
        return np.exp(-rate * maturity)

    def get_zc_rate(self, discount_factor: float, maturity: float) -> float:
        """
        Returns Zero-Coupon rate based on the maturity and discount factor.
        
        Input: Discount factor, maturity
        """
        return -np.log(discount_factor)/maturity 

    def get_ytm(self, market_price: float, maturity: float) -> float:
        """
        Returns actual yield to maturity from market price and maturity.
        
        Input: price, maturity.
        """
        if maturity == 0:
            raise ValueError("Maturity cannot be zero when computing YTM.")
        return (self.__nominal/market_price)**(1/maturity) - 1

    def get_duration_macaulay(self, maturity: float) -> float:
        """
        Returns Macaulay duration (=maturity for a ZC)
        
        Input: maturity
        """
        return maturity

    def get_modified_duration(self, market_price: float, maturity: float) -> float:
        """
        Returns Modified Duration.
        
        Input: market price, maturity
        """
        return self.get_duration_macaulay(maturity)/(1+self.get_ytm(market_price, maturity))

    def get_sensitivity(self, new_rate: float, maturity: float) -> float:
        """
        Returns sensitivity from a zero coupon bond
        
        Input: new_rate, maturity
        """
        return self.get_duration_macaulay(maturity)/(1+new_rate)

    def get_convexity(self, maturity: float, market_price: float = None, discount_factor: float=None) -> float:
        """
        Computes the convexity of a zero-coupon bond. Works for issued ones, and non issued ones.
        
        Input:
        - T (float): Time to maturity (in years)
        - market_price (float, optional): Market price of the bond. If None, we condisder the case non issued ZC (NEED DF).
        - discount factor (float, optional): If None, we consider an issued ZC (NEED market price)
        
        Returns: float: Convexity of the zero-coupon bond.
        """
        if market_price is not None and discount_factor is None:
            ytm = self.get_ytm(market_price, maturity)
            return (maturity*(maturity+1)*self.__nominal)/(market_price*((1+ytm)**(2+maturity)))
        elif discount_factor is not None and market_price is None:
            market_price = self.get_npv_zc_from_df(discount_factor)
            ytm = self.get_ytm(market_price, maturity)
            return (maturity*(maturity+1)*self.__nominal)/(market_price*((1+ytm)**(2+maturity)))
        else:
            raise ValueError(f"Incorrect input, we need discount factor OR market price: DF = {discount_factor} and market price = {market_price}.")
      
#-------------------------------------------------------------------------------------------------------
#----------------------------Classes de produits généralistes en fixed-income---------------------------
#-------------------------------------------------------------------------------------------------------
#____________________________Classe abstraite pour les produits d'Income________________________________
class FixedIncomeProduct(ABC):
    """
    Abstract class for fixed-income products:

    Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)

    Abstract class to build the different types of legs for fixed income instruments.
    For fixed income leg, rate_curve will be a flat rate curve.
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION,
                 discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        
        self._rate_curve=rate_curve
        self._start_date=start_date
        self._end_date=end_date
        self._paiement_freq=paiement_freq
        self._currency=currency
        self._day_count=day_count
        self._discounting_curve = discounting_curve
        self._rolling_conv=rolling_conv
        self._notional = notional
        self._format = format
        self._cashflows = {}
        self._cashflows_cap ={}
        self._cashflows_floor ={}
        self._cashflows_r = {}
        self._cashflows_cap_r ={}
        self._cashflows_floor_r ={}
        self._exchange_notional = exchange_notional
        self._spread = spread
        self._interpol = interpol
        self._paiments_schedule = \
            PaymentScheduleHandler(self._start_date, self._end_date,
            self._paiement_freq, self._format).build_schedule(\
            convention=self._day_count, rolling_convention=self._rolling_conv, market=\
            utils.get_market(currency=self._currency))

    @abstractmethod
    def calculate_npv(self,cashflows) -> float:
        """
        Returns the product NPV as float
        """
        return sum(entry["NPV"] for entry in cashflows.values())
    
    @abstractmethod
    def calculate_duration(self) -> float:
        """
        Returns duration of the product
        """
        duration_ti = sum(value["NPV"] * key for key, value in self._cashflows.items())
        return duration_ti / self.calculate_npv(self._cashflows)

    @abstractmethod
    def calculate_sensitivity(self) -> float:
        """
        Returns sensitivity of the product
        """
        pass

    @abstractmethod
    def calculate_convexity(self) -> float:
        """
        Returns convexity of the product
        """
        pass

    @abstractmethod
    def calculate_pv01(self) -> float:
        """
        Returns pov01 of the product
        """
        return sum(entry["PV01"] for entry in self._cashflows.values())

#Fixed Leg Income product
class FixedLeg(FixedIncomeProduct):
    """
    Class pour une leg fixe, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et fixed bonds.

    Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)

    Returns: class with functions of NPV, duration, convexity, pv01, etc.
    For fixed income leg, rate_curve will be a flat rate curve.
    For bonds, notional exchange will be True in build_cashflows.
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str, paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, shift:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, shift, format, interpol, exchange_notional)
    
        self._rates_c = self._rate_curve.create_product_rate_curve(self._paiments_schedule, "Flat")

        if discounting_curve is None:
            self._discounting_c=self._rate_curve
        else:
            self._discounting_c=discounting_curve

        self._discountings=self._discounting_c.create_product_rate_curve(self._paiments_schedule, TYPE_INTERPOL)

        self._ZC = ZCBond(self._notional)
        self._rate_dict = dict(zip(self._rates_c["Year_fraction"], self._rates_c["Rate"]))
        self._discount_dict = dict(zip(self._discountings["Year_fraction"], self._ZC.get_discount_factor_from_zcrate(self._discountings["Rate"]/100, self._discountings["Year_fraction"])))

        self.build_cashflows()
        self.build_cashflows_npv()
        pass

    def calculate_npv(self,cashflows) -> float:
        """
        Calculate the NPV of the fixed leg.
        """
        return super().calculate_npv(cashflows)

    def calculate_duration(self) -> float:
        """
        Calculate the duration of the fixed leg.
        """
        return super().calculate_duration()

    def calculate_sensitivity(self, shift:dict=None) -> float:
        """
        Calculate the sensitivity of the fixed leg.

        Input:
        - shift (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift = dict(zip(self._paiments_schedule, s))
        shifted_curve = self._discounting_c.deep_copy()
        shifted_curve.shift_curve(shift, self._interpol)
        shift_fixed_leg = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        shift_fixed_leg.calculate_npv(self._cashflows)
        return shift_fixed_leg.calculate_npv(shift_fixed_leg._cashflows) - self.calculate_npv(self._cashflows)

    def calculate_convexity(self, shift:dict=None) -> float:
        """
        Calculate the convexity of the fixed leg.

        Input:
        - shift (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps (0.01 input).
        """
        if shift is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift = dict(zip(self._paiments_schedule, s))
        
        shifted_pos_curve = self._discounting_c.deep_copy()
        shifted_pos_curve.shift_curve(shift, self._interpol)
        shift_leg_pos = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_pos_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        
        neg_shift = {key: -value for key, value in shift.items()}
        shifted_neg_curve = self._discounting_c.deep_copy()
        shifted_neg_curve.shift_curve(neg_shift, self._interpol)
        shift_leg_neg = FixedLeg(self._rate_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_neg_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        return sum((shift_leg_pos.calculate_npv(shift_leg_pos._cashflows) + shift_leg_neg.calculate_npv(shift_leg_neg._cashflows) - 2 * self.calculate_npv(self._cashflows)) /
            ((shift[t]/100 ** 2) * self.calculate_npv(self._cashflows)) for t in self._paiments_schedule)
    
    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the fixed leg.
        """
        return super().calculate_pv01()
    
    def build_cashflows_npv(self) -> dict:
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_r.items()
        }
        self._cashflows = discounted_cashflows
        pass

    def build_cashflows(self) -> dict:
        """
        Build the paiements schedule for the fixed leg (RAW).
        Input:
        - exchange_notionnal (string, optional, equal to False if not provided), provide True for bonds.
        """
        for i in range(len(self._paiments_schedule)-1):
            date = self._paiments_schedule[i]
            if date == self._paiments_schedule[0]:
                npv = self._notional * (self._rate_dict[date]+self._spread)/100 * date
                pv01 = self._notional * 1/10000 * date
                self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
            elif date != self._paiments_schedule[-1] and date!= self._paiments_schedule[0]:
                npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1])
                pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
            else:
                if self._exchange_notional == True:
                    npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1]) + self._notional
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
                else:
                    npv = self._notional * (self._spread+self._rate_dict[date])/100 * (date-self._paiments_schedule[i-1])
                    pv01 = self._notional * 1/10000  * (date-self._paiments_schedule[i-1])
                    self._cashflows_r[date] = {"NPV": npv, "PV01": pv01}
        pass

    def calculate_yield(self, market_price:float) -> float:
        """
        Calculate the yield of the fixed leg.
        """
        return utils.calculate_yield(self._cashflows_r, market_price)

#Float Leg Income product
class FloatLeg(FixedIncomeProduct):
    """
    Class pour une leg flottante, on va pouvoir calculer le npv, duration, convexity, pv01, etc.
    Utilisée pour les swaps et FRNs.

    Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)
    """
    def __init__(self, rate_curve: Rates_curve, start_date:str, end_date:str,
                 paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION,
                 discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)       
        
        self._rates_c = self._rate_curve.create_product_rate_curve(self._paiments_schedule, interpol)
        if discounting_curve is None:
            self._discounting_c=self._rate_curve
        else:
            self._discounting_c=discounting_curve

        self._discountings=self._discounting_c.create_product_rate_curve(self._paiments_schedule, interpol)
        
        self._ZC = ZCBond(self._notional)
        self._rate_dict = dict(zip(self._rates_c["Year_fraction"], self._rates_c["Forward_rate"]))
        self._discount_dict = dict(zip(self._discountings["Year_fraction"], self._ZC.get_discount_factor_from_zcrate(self._discountings["Rate"]/100, self._discountings["Year_fraction"])))
        self.build_cashflows(self._rate_dict,100, self._cashflows_r)
        self.build_cashflows_npv()
        pass
    
    def calculate_npv(self,cashflows) -> float:
        """
        Calculate the NPV of the float leg.
        """
        return super().calculate_npv(cashflows)
    
    def calculate_duration(self) -> float:
        """
        Calculate the duration of the float leg.
        """
        return super().calculate_duration()
    
    def calculate_sensitivity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the sensitivity of the float leg.

        Input:
        - shift_fw (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift_fw is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift_fw = dict(zip(self._paiments_schedule, s))

        if shift_discounting is None:
            shift_discounting = shift_fw

        shifted_fw_curve = self._rate_curve.deep_copy()
        shifted_fw_curve.shift_curve(shift_fw, self._interpol)

        shifted_discounting_curve = self._discounting_c.deep_copy()
        shifted_discounting_curve.shift_curve(shift_discounting, self._interpol)

        shift_fixed_leg = FloatLeg(shifted_fw_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_discounting_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
        shift_fixed_leg.calculate_npv(self._cashflows)
        return shift_fixed_leg.calculate_npv(shift_fixed_leg._cashflows) - self.calculate_npv(self._cashflows)

    def calculate_convexity(self, shift_fw:dict=None, shift_discounting:dict=None) -> float:
        """
        Calculate the convexity of the float leg.

        Input:
        - shift_fw (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        - shift_discounting (dict, optional): dictionnary of shift for each date, if not given -> linear shift of 1bps.
        """
        if shift_fw is None:
            s = np.ones(len(self._paiments_schedule)) * BASE_SHIFT
            shift_fw = dict(zip(self._paiments_schedule, s))

        if shift_discounting is None:
            shift_discounting = shift_fw
        
        def get_npv(shift_fw, shift_ds):
            shifted_fw_curve = self._rate_curve.deep_copy()
            shifted_fw_curve.shift_curve(shift_fw, self._interpol)
            shifted_discounting_curve = self._discounting_c.deep_copy()
            shifted_discounting_curve.shift_curve(shift_ds, self._interpol)
            shift_leg = FloatLeg(shifted_fw_curve, self._start_date, self._end_date, self._paiement_freq, self._currency, self._day_count, self._rolling_conv, shifted_discounting_curve, self._notional, self._spread, self._format, self._interpol, self._exchange_notional)
            return shift_leg.calculate_npv(self._cashflows)

        #Initial NPV:
        npv_0 = self.calculate_npv(self._cashflows)

        #Shift in the same direction of both curves:
        npv_pp = get_npv(shift_fw, shift_discounting)
        npv_mm = get_npv({key: -value for key, value in shift_fw.items()}, {key: -value for key, value in shift_discounting.items()})
        
        #Shift in the opposite direction of both curves:
        npv_pm = get_npv(shift_fw, {key: -value for key, value in shift_discounting.items()})
        npv_mp = get_npv({key: -value for key, value in shift_fw.items()}, shift_discounting)

        #Shift for one and not the other:
        npv_p0 = get_npv(shift_fw, {t: 0 for t in shift_discounting})
        npv_0p = get_npv({t: 0 for t in shift_fw}, shift_discounting)
        npv_m0 = get_npv({key: -value for key, value in shift_fw.items()}, {t: 0 for t in shift_discounting})
        npv_0m = get_npv({t: 0 for t in shift_fw}, {key: -value for key, value in shift_discounting.items()})

        sum_npvs = npv_pp + npv_mm - npv_p0 - npv_0p + npv_pm + npv_mp + 2 * npv_0 - npv_m0 - npv_0m

        vector_f = np.array([value for value in shift_fw.values()])
        vector_d = np.array([value for value in shift_discounting.values()])
        d = np.dot(vector_f, vector_d) * npv_0

        return sum_npvs / (d *10000)

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the float leg.
        """
        return super().calculate_pv01()
    
    def build_cashflows(self,dict,pourcentage,dict_result) -> dict:
        """
        Build the paiements schedule for the fixed leg.
        Input:
        - exchange_notionnal (string, optional, equal to False if not provided), provide True for bonds.
        """

        for i in range(len(self._paiments_schedule)-1):
            date = self._paiments_schedule[i]
            if date == self._paiments_schedule[0]:
                npv = self._notional * (dict[date]+self._spread)/pourcentage * date
                pv01 = self._notional * 1/10000 * date
                dict_result[date] = {"NPV": npv, "PV01": pv01}
            elif date != self._paiments_schedule[-1] and date!= self._paiments_schedule[0]:
                npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1])
                pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                dict_result[date] = {"NPV": npv, "PV01": pv01}
            else: 
                if self._exchange_notional == True:
                    npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1]) + self._notional
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    dict_result[date] = {"NPV": npv, "PV01": pv01}
                else:
                    npv = self._notional * (dict[date]+self._spread)/pourcentage * (date-self._paiments_schedule[i-1])
                    pv01 = self._notional * 1/10000 * (date-self._paiments_schedule[i-1])
                    dict_result[date] = {"NPV": npv, "PV01": pv01}
        pass

    def build_cashflows_npv(self) -> dict:
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_r.items()
        }
        self._cashflows = discounted_cashflows
        pass

    def calculate_yield(self, market_price:float) -> float:
        """
        Calculate the yield of the float leg.
        """
        return utils.calculate_yield(self._cashflows_r, market_price)

    def cap_value(self, cap_strike:float,sigma:float) -> float:
        """
        Calculate the cap value of the float leg.

        Input:
        - cap (float): cap value
        - sigma (float): volatility of the forward rate

        """
        df_cap= pd.DataFrame()
        df_cap["Log"] = self._rates_c["Forward_rate"].apply(lambda x: np.log((x/100)/cap_strike))
        df_cap["vol"] = (0.5*sigma**2)*self._rates_c["Year_fraction"]
        df_cap["Actu"] = sigma*np.sqrt(self._rates_c["Year_fraction"])
        df_cap["d1"] = (df_cap["Log"]+df_cap["vol"])/df_cap["Actu"]
        df_cap["d2"] = df_cap["d1"]-df_cap["Actu"]
        df_cap["value"] = self._rates_c["Forward_rate"]/100*norm.cdf(df_cap["d1"])-cap_strike*norm.cdf(df_cap["d2"])   
        self._cap_rate_dict = dict(zip(self._rates_c["Year_fraction"],  df_cap["value"]))
        self.build_cashflows(self._cap_rate_dict,1, self._cashflows_cap_r)
        self.build_cashflow_cap_npv()
        pass

    def build_cashflow_cap_npv(self):
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """  
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_cap_r.items()
        }
        self._cashflows_cap = discounted_cashflows
        pass

    def floor_value(self, floor_strike:float,sigma:float) -> float:
        """
        Calculate the cap value of the float leg.

        Input:
        - cap (float): cap value
        - sigma (float): volatility of the forward rate

        """
        df_floor= pd.DataFrame()
        df_floor["Log"] = self._rates_c["Forward_rate"].apply(lambda x: np.log((x/100)/floor_strike))
        df_floor["vol"] = (0.5*sigma**2)*self._rates_c["Year_fraction"]
        df_floor["Actu"] = sigma*np.sqrt(self._rates_c["Year_fraction"])
        df_floor["d1"] = (df_floor["Log"]+df_floor["vol"])/df_floor["Actu"]
        df_floor["d2"] = df_floor["d1"]-df_floor["Actu"]
        df_floor["value"] = floor_strike*norm.cdf(-1*df_floor["d2"]) - self._rates_c["Forward_rate"]/100*norm.cdf(-1*df_floor["d1"])   
        self._floor_rate_dict = dict(zip(self._rates_c["Year_fraction"],  df_floor["value"]))
        self.build_cashflows(self._floor_rate_dict,1, self._cashflows_floor_r)
        self.build_cashflow_floor_npv()
        pass

    def build_cashflow_floor_npv(self):
        """
        Apply discount factors to NPV and PV01 in a cashflow dictionary.
        
        :return: New dictionary with discounted "NPV" and "PV01".
        """  
        discounted_cashflows = {
            t: {
                "NPV": cf["NPV"] * self._discount_dict.get(t, 1),
                "PV01": cf["PV01"] * self._discount_dict.get(t, 1)
            }
            for t, cf in self._cashflows_floor_r.items()
        }
        self._cashflows_floor = discounted_cashflows
        pass

#Class Swap
class Swap(FixedIncomeProduct):
    """
    Class pour un swap classique, on va pouvoir trouver le taux d'un swap

    - Float and Fixed leg nous permettent de créer un swap.
    - La classe permet d'utiliser les fonctions de FixedLeg et FloatLeg pour trouver le taux d'un swap. 

       Input:
    - forward rates curve (dict, non optional)
    - start date (string, non optional)
    - end date (string, non optional)
    - paiments frequency (string, non optional)
    - day count convention (string, optional, equal to 30/360 if not provided)
    - rolling convention (string, optional, equal to Modified Following if not provided)
    - discounting curve to discount with a different curve than the forward rates curve (dict, optional)
    - notional (float, optional, will quote in percent if not provided)
    """
    def __init__ (self, rate_curve: Rates_curve, start_date:str, end_date:str, paiement_freq:str, currency:str, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, discounting_curve:Rates_curve=None, notional:float=BASE_NOTIONAL, spread:float=0, format:str=FORMAT_DATE, interpol: str=TYPE_INTERPOL, exchange_notional: str=EXCHANGE_NOTIONAL) -> None:
        super().__init__(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)

        self.float_leg = FloatLeg(rate_curve, start_date, end_date, paiement_freq, currency, day_count, rolling_conv, discounting_curve, notional, spread, format, interpol, exchange_notional)

    def calculate_fixed_rate(self) -> float:
        """
        Calculate the fixed rate of the swap and initialize the FixedLeg.
        """
        # Calculer la NPV et PV01 de la jambe flottante
        float_npv = self.float_leg.calculate_npv(self.float_leg._cashflows)
        float_pv01 = self.float_leg.calculate_pv01()

        # Calculer le taux fixe
        fixed_rate = (float_npv / float_pv01) / 10000

        # Créer une copie de la courbe de taux avec le taux fixe
        self._rate_curve_fixed = self._rate_curve.deep_copy(fixed_rate)

        # Initialiser la jambe fixe
        self.fixed_leg = FixedLeg(
            self._rate_curve_fixed,
            self._start_date,
            self._end_date,
            self._paiement_freq,
            self._currency,
            self._day_count,
            self._rolling_conv,
            self._discounting_curve,
            self._notional,
            self._spread,
            self._format,
            "Flat",
            self._exchange_notional
        )

        return fixed_rate
    
    def calculate_collar(self, cap_strike:float, floor_strike:float, sigma:float) -> float:
        """
        Calculate the collar value of the swap.

        Input:
        - cap (float): cap value
        - floor (float): floor value
        - sigma (float): volatility of the forward rate
        """
        self.float_leg.cap_value(cap_strike, sigma)
        self.float_leg.floor_value(floor_strike, sigma)

        # Calculate the collar value
        collar_value = self.float_leg.calculate_npv(self.float_leg._cashflows_cap) - self.float_leg.calculate_npv(self.float_leg._cashflows_floor)
        
        return collar_value
    
    def calculate_npv(self) -> float:
        """
        Calculate the NPV of the swap.
        """
        return self.float_leg.calculate_npv() - self.fixed_leg.calculate_npv()

    def calculate_duration(self) -> float:
        """
        Calculate the duration of the swap.
        """
        return (self.float_leg.calculate_duration() + self.fixed_leg.calculate_duration()) / 2

    def calculate_sensitivity(self) -> float:
        """
        Calculate the sensitivity of the swap.
        """
        return self.float_leg.calculate_sensitivity() - self.fixed_leg.calculate_sensitivity()

    def calculate_convexity(self) -> float:
        """
        Calculate the convexity of the swap.
        """
        return self.float_leg.calculate_convexity() - self.fixed_leg.calculate_convexity()

    def calculate_pv01(self) -> float:
        """
        Calculate the PV01 of the swap.
        """
        return self.float_leg.calculate_pv01() - self.fixed_leg.calculate_pv01()

#-------------------------------------------------------------------------------------------------------
#----------------------------Classes de produits généralistes en equity derivatives---------------------
#-------------------------------------------------------------------------------------------------------
#____________________________Classe abstraite pour les produits d'Equity________________________________
class EQDProduct(ABC):
    """"
    Abstract class for equity derivatives products.

    Input:
    -Type (Enum, optional) (call / put)
    -Spot (float, optional)
    -Strike (float, optional)
    -Rate (float, optional)
    
    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    
    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, strike:float=BASE_STRIKE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, price:float=None, periodicity:str=None) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._type = type
        self._strike = strike
        self._rate = rate
        self._day_count = day_count
        self._rolling_conv = rolling_conv
        self._notional = notional
    
        self._format=format_date
        self._currency=currency
        self._price=price
        if periodicity is None:
            self._periodicity="none"
        else:
            self._periodicity=periodicity

        self._paiments_schedule = \
            PaymentScheduleHandler(self._start_date, self._end_date,
            self._periodicity, self._format).build_schedule(\
            convention=self._day_count, rolling_convention=self._rolling_conv, market=\
            utils.get_market(currency=self._currency))
        
        self.T = self._paiments_schedule[-1]
        pass

    @property
    @abstractmethod
    def npv(self) -> float:
        """
        Calculate the NPV of the equity derivative product.
        """
        pass

    @property
    @abstractmethod
    def payoff(self) -> float:
        """
        Calculate the payoff of the equity derivative product.
        """
        pass

#Class for vanilla equities options:
class VanillaOption(EQDProduct):
    """"
    Class for Vanilla Option.

    Input:
    -Type (Enum, optional) (call / put)
    -Spot (float, optional)
    -Strike (float, optional)
    -Rate (float, optional)
    
    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    
    Returns:
    - NPV
    - Greeks: Delta / Gamma / Rho / Theta / Vega
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, strike:float=BASE_STRIKE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, div_rate:str=BASE_DIV_RATE, price:float=None, volume:float=None) -> None:
        super().__init__(start_date, end_date, type, strike, rate, day_count, rolling_conv, notional, format_date, currency, price)
        self._div_rate=div_rate
        self._volume=volume
        pass

    def npv(self, spot):
        """
        Calculate the NPV of the equity derivative product.
        """
        return self.payoff(spot) * self._notional * np.exp(-self._rate * self.T)
    
    def payoff(self, spot):
        """
        Calculate the payoff of the equity derivative product.
        """
        
        if self._type == OptionType.CALL:
            return max(spot - self._strike, 0)
        elif self._type == OptionType.PUT:
            return max(self._strike - spot, 0)
        else:
            ValueError("Option type not recognized !")
            pass
        
    def __deep_copy__(self):
        return VanillaOption(self._start_date, self._end_date, self._type, self._strike, self._rate, self._day_count, self._rolling_conv, self._notional, self._format, self._currency, self._div_rate, self._price, self._volume)

    def __rebuild__(self):
        self.__init__(
            start_date=self._start_date, 
            end_date=self._end_date, 
            type=self._type, 
            strike=self._strike, 
            rate=self._rate, 
            day_count=self._day_count, 
            rolling_conv=self._rolling_conv, 
            notional=self._notional, 
            format_date=self._format, 
            currency=self._currency, 
            div_rate=self._div_rate, 
            price=self._price, 
            volume=self._volume
        )
        pass

#Class for barriers options
class BarrierOption(EQDProduct):
    """
    Class for Barrier Option.

    Input:
    -Type (Enum, optional)
    -Barrier type (Enum, non optional)
    -Spot (float, optional)
    -Strike (float, optional)
    -Barrier Strike (float, optional)
    -Rate (float, optional)

    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    """

    def __init__(self, start_date:str, end_date:str, type:str=BarrierType.CALL_UP_OUT,\
                 strike:float=BASE_STRIKE, barrier_strike:float=BASE_STRIKE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT,\
                 rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE,\
                 currency:str=BASE_CURRENCY, div_rate:str=BASE_DIV_RATE, price:float=None, volume:float=None) -> None:
        
        super().__init__(start_date, end_date, type, strike, rate, day_count, rolling_conv, notional, format_date, currency, price)
        self._div_rate=div_rate
        self._volume=volume
        self._barrier_strike=barrier_strike
        pass

    def npv(self, spot):
        """
        Calculate the NPV of the equity derivative product.
        """
        return self.payoff(spot) * self._notional * np.exp(-self._rate * self.T)
    
    def payoff(self, spot):
        """
        Calculate the payoff of the equity derivative product.
        """
        if self._type == BarrierType.CALL_UP_IN:
            if spot > self._barrier_strike:
                return max(spot - self._strike, 0)
            else:
                return 0
            
        elif self._type == BarrierType.CALL_UP_OUT:
            if spot < self._barrier_strike:
                return max(spot - self._strike, 0)
            else:
                return 0

        elif self._type == BarrierType.CALL_DOWN_IN:
            if spot < self._barrier_strike:
                return max(spot - self._strike, 0)
            else:
                return 0
        
        elif self._type == BarrierType.CALL_DOWN_OUT:
            if spot > self._barrier_strike:
                return max(spot - self._strike, 0)
            else:
                return 0

        elif self._type == BarrierType.PUT_UP_IN:
            if spot > self._barrier_strike:
                return max(self._strike - spot, 0)
            else:
                return 0

        elif self._type == BarrierType.PUT_UP_OUT:
            if spot < self._barrier_strike:
                return max(self._strike - spot, 0)
            else:
                return 0

        elif self._type == BarrierType.PUT_DOWN_IN:
            if spot < self._barrier_strike:
                return max(self._strike - spot, 0)
            else:
                return 0
        
        elif self._type == BarrierType.PUT_DOWN_OUT:
            if spot > self._barrier_strike:
                return max(self._strike - spot, 0)
            else:
                return 0
        else:
            ValueError("Barrier type not recognized !")
            pass

    def __deep_copy__(self):
        return BarrierOption(self._start_date, self._end_date, self._type, self._strike, self._barrier_strike, self._rate, self._day_count, self._rolling_conv, self._notional, self._format, self._currency, self._div_rate, self._price, self._volume)

    def __rebuild__(self):
        self.__init__(
        start_date=self._start_date,
        end_date=self._end_date, 
        type=self._type, 
        strike=self._strike, 
        barrier_strike=self._barrier_strike, 
        rate=self._rate, 
        day_count=self._day_count, 
        rolling_conv=self._rolling_conv, 
        notional=self._notional, 
        format_date=self._format, 
        currency=self._currency, 
        div_rate=self._div_rate, 
        price=self._price, 
        volume=self._volume
        )
        pass

#Vanilla Autocalls
class Autocalls(EQDProduct):
    """
    Class for Autocalls Options (Only one autocall barriers, paying coupons).

    Input:
    -Type (Enum, optional)
    -Spot (float, optional)
    -Strike (float, optional) Given on scale 1 -> 70% = 0.7
    -Rate (float, optional)

    -date format (string, optional)
    -currency (string, optional)
    -start date (string, optional)
    -end date (string, optional) / options dates
    -day count convention (string, optional, equal to 30/360 if not provided)
    -rolling convention (string, optional, equal to Modified Following if not provided)
    -notional (float, optional, will quote in percent if not provided) / nb underlying shares / equities
    """
    def __init__(self, start_date:str, end_date:str, type:str=AutocallsType.AUTOCALL,\
                strike:float=1, final_strike:float=None, coupon:float=None, coupon_strike:float=None, protection:float=0, memory:bool=True, type_opt:str=Types.AMERICAN,
                rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT,\
                rolling_conv:str=ROLLING_CONVENTION, frequency:str=None, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE,\
                currency:str=BASE_CURRENCY, div_rate:str=BASE_DIV_RATE, price:float=None, volume:float=None) -> None:
    
        super().__init__(start_date, end_date, type, strike, rate, day_count, rolling_conv, notional, format_date, currency, price, frequency)
        self._div_rate=div_rate
        self._volume=volume
        self._type_opt = type_opt
        self._barriers = None
        self._coupon_strike = coupon_strike
        self._protection = protection
        self._memory = memory

        if final_strike is None:
            self._final_strike = self._strike
        else:
            self._final_strike = final_strike   
        
        if frequency is None:
            self._type_opt = Types.AMERICAN
        pass

        if coupon is None:
            self._coupon = BASE_RATE
        else:
            self._coupon = coupon
        self._spot = None
        self._paths=None

    def set_paths(self, paths:dict):
        """
        Must be called before doing any calculation.
        """
        self._paths = paths
        self._barriers=self._create_barrier()
        pass

    def price(self):
        """
        Launch the Autocall pricing.
        """
        cp = self._coupon
        npv, payoff = self.npv
        #df_payoff = pd.DataFrame(payoff)
        #df_payoff.to_excel("test_payoff.xlsx")
        call_prob = self._call_probability_curve()
        par_cpn = self._calculate_par_coupon()
        self._coupon=cp
        return npv, payoff, par_cpn, call_prob

    @property
    def npv(self):
        """
        Calculate the NPV of the EQD product.
        """
        payoff = self.payoff
        dfs = np.exp(-self._rate * self._paths['time'])
        discounted_payoffs = payoff * dfs[np.newaxis, :]
        total_npv = np.mean(np.sum(discounted_payoffs, axis=1))
        return total_npv, payoff
    
    @property
    def payoff(self):
        """
        Calculate all the payoffs of the EQD product.
        """
        time = self._paths['time']
        spots = self._paths['Spots']
        n_paths, n_steps = spots.shape

        if self._type_opt == Types.AMERICAN:
            cpn = self._coupon * (self.T / (n_steps - 1)) * self._notional
        else:
            cpn = self._coupon * (self.T/len(self._paiments_schedule)) * self._notional
            step_indices = [np.argmin(np.abs(time - sched)) for sched in self._paiments_schedule]

        coupons = np.zeros_like(spots)
        memory = np.zeros(n_paths) if self._memory and self._type == AutocallsType.PHOENIX else None
        payoffs = np.zeros_like(spots)
        flag = np.zeros(n_paths, dtype=bool)
        self.call_counts = np.zeros(n_steps, dtype=int)

        if self._type_opt == Types.AMERICAN:
            for i in range(1,n_steps):
                barrier = np.ones(n_paths) * self._barriers[i]
                c_barrier = np.ones(n_paths) * self._coupon_strike[i]
                act_spots = spots[:,i]

                eligible_for_coupon = (act_spots >= c_barrier) & (~flag)
                if self._type == AutocallsType.PHOENIX:
                    if self._memory:
                        coupons[:, i][eligible_for_coupon] = cpn + memory[eligible_for_coupon]
                        memory[eligible_for_coupon] = 0
                        memory[~eligible_for_coupon] += cpn
                    else:
                        coupons[:, i][eligible_for_coupon] = cpn                
                    payoffs[:, i] = coupons[:,i]
                else:
                    eligible_for_coupon = (act_spots >= c_barrier) & (~flag)
                    coupons[eligible_for_coupon] += cpn
                
                redeem = (act_spots >= barrier) & (~flag)
                if self._type == AutocallsType.PHOENIX:
                    payoffs[redeem, i] += self._notional
                    coupons[redeem, i+1:] = 0
                else:
                    coupons[redeem] = cpn * i
                    payoffs[redeem, i] = self._notional + coupons[redeem, i]

                flag[redeem] = True
                self.call_counts[i] = np.sum(redeem)
            
            final_spots=spots[:,-1]
            final_barrier = self._barriers[-1]
            no_redeemed= ~flag

            protected = final_spots >= self._protection
            non_protected = final_spots < self._protection
            if self._type == AutocallsType.PHOENIX:
                payoffs[:, -1][no_redeemed & protected] = self._notional
                payoffs[:, -1][no_redeemed & non_protected] = self._notional *(final_spots[no_redeemed & non_protected] / final_barrier)
                payoffs[:, -1][no_redeemed] += coupons[:,-1][no_redeemed]
            else:
                payoffs[:, -1][no_redeemed & protected] = self._notional + cpn * (n_steps - 1)
                payoffs[:, -1][no_redeemed & non_protected] = self._notional * (final_spots[no_redeemed & non_protected] / final_barrier)
            return payoffs
        
        elif self._type_opt == Types.EUROPEAN:
            for j, step in enumerate(step_indices):
                barrier = np.ones(n_paths) * self._barriers[step]
                c_barrier = np.ones(n_paths) * self._coupon_strike[step]
                act_spots = spots[:,step]

                eligible_for_coupon = (act_spots >= c_barrier) & (~flag)
                if self._type == AutocallsType.PHOENIX:
                    if self._memory:
                        coupons[:, step][eligible_for_coupon] = cpn + memory[eligible_for_coupon]
                        memory[eligible_for_coupon] = 0
                        memory[~eligible_for_coupon] += cpn
                    else:
                        coupons[:, step][eligible_for_coupon] = cpn                
                    payoffs[:, step] = coupons[:,step]

                else:
                    eligible_for_coupon = (act_spots >= c_barrier) & (~flag)
                    coupons[eligible_for_coupon] += cpn

                redeem = (act_spots >= barrier) & (~flag)
                if self._type == AutocallsType.PHOENIX:
                    payoffs[redeem, step] += self._notional
                    coupons[redeem, step+1:] = 0
                else:
                    coupons[redeem] = cpn * (j+1)
                    payoffs[redeem, step] = self._notional + coupons[redeem, step]
                
                flag[redeem] = True
                self.call_counts[step] = np.sum(redeem)

            final_spots=spots[:,-1]
            final_barrier = self._barriers[-1]
            no_redeemed= ~flag

            protected = final_spots >= self._protection
            non_protected = final_spots < self._protection
            if self._type == AutocallsType.PHOENIX:
                payoffs[:, -1][no_redeemed & protected] = self._notional
                payoffs[:, -1][no_redeemed & non_protected] = self._notional *(final_spots[no_redeemed & non_protected] / final_barrier)
                payoffs[:, -1][no_redeemed] += coupons[:,-1][no_redeemed]
            else:
                payoffs[:, -1][no_redeemed & protected] = self._notional + cpn * (n_steps - 1)
                payoffs[:, -1][no_redeemed & non_protected] = self._notional * (final_spots[no_redeemed & non_protected] / final_barrier)
            return payoffs
        else:
            ValueError(f"Type of exercize: {self._type_opt} not recognize!")
        pass

    def _create_barrier(self):
        """
        Create the barriers dict to populate with the possible dynamic barriers.
        """
        times=self._paths['time'].tolist()
        self._spot=self._paths['Spots'][0][0]
        steps=len(times)
        barriers=np.linspace(self._strike, self._final_strike, steps+1) * self._spot

        n_paths = len(self._paths['Spots'])
        if self._coupon_strike is None:
            self._coupon_strike = np.array(barriers)
        else:
            self._coupon_strike = np.array(np.ones(steps) * self._spot * self._coupon_strike)

        if self._protection is None:
            self._protection = np.array(np.ones(n_paths)* barriers[-1])
        else: 
            self._protection = np.array(np.ones(n_paths) * self._spot * self._protection)
        return list(barriers)
    
    def _calculate_par_coupon(self):
        """
        Find the coupon that matches NPV = 0 (or close to).
        """
        def objective(coupon):
            mem_cpn = self._coupon
            self._coupon = coupon
            npv, payoff = self.npv
            self._coupon = mem_cpn
            return (self._notional-npv)**2

        result = minimize(objective, self._coupon, method=SOLVER_METHOD, bounds=[(0, 1)])
        if result.success:
            return result.x[0]
        else:
            raise ValueError("Coupon calculation failed.")       

    def _call_probability_distribution(self):
        """Return a dictionary of call step -> probability of being autocalled."""
        n_paths = self._paths['Spots'].shape[0]
        return {i: self.call_counts[i] / n_paths for i in range(len(self.call_counts)) if self.call_counts[i] > 0}

    def _call_probability_curve(self):
        """
        Returns a list of (year_fraction, cumulative_probability) sorted by time step.
        """
        call_prob = self._call_probability_distribution()
        time_grid = self._paths['time']
        
        cumulative = 0.0
        result = []

        for step in sorted(call_prob.keys()):
            cumulative += call_prob[step]
            year_fraction = time_grid[step]
            result.append((year_fraction, cumulative))

        return result

    def __deep_copy__(self):
        return Autocalls(start_date=self._start_date, 
                         end_date=self._end_date, 
                         type=self._type,
                         strike=self._strike, 
                         final_strike=self._final_strike, 
                         coupon=self._coupon, 
                         coupon_strike=self._coupon_strike, 
                         protection=self._protection, 
                         memory=self._memory, 
                         type_opt=self._type_opt, 
                         rate=self._rate, 
                         day_count=self._day_count, 
                         rolling_conv=self._rolling_conv, 
                         frequency=self._periodicity, 
                         notional=self._notional, 
                         format_date=self._format, 
                         currency=self._currency, 
                         div_rate=self._div_rate, 
                         price=self._price, 
                         volume=self._volume)
    pass

    def __rebuild__(self):
        self.__init__(
        start_date=self._start_date, 
        end_date=self._end_date, 
        type=self._type, 
        strike=self._strike, 
        final_strike=self._final_strike, 
        coupon=self._coupon, 
        coupon_strike=self._coupon_strike, 
        protection=self._protection, 
        memory=self._memory, 
        type_opt=self._type_opt, 
        rate=self._rate, 
        day_count=self._day_count, 
        rolling_conv=self._rolling_conv, 
        frequency=self._periodicity, 
        notional=self._notional, 
        format_date=self._format, 
        currency=self._currency, 
        div_rate=self._div_rate, 
        price=self._price, 
        volume=self._volume
        )
        pass

#Classe gestion du marché des options: Ne pas bouger de fichier, obligé d'être ici pour éviter les appels en ronds.
class OptionMarket:
    """
    Class to handle the dataset of options, select appropriate ones for volatility models/parameters calibration.

    Input:
    - Filename / path for the options CSV dataset
    """
    def __init__(self, filename_options:str, filename_uderlying:str) -> None:
        self._filename = filename_options
        self._filename_underlying = filename_uderlying
        self._df_asset = pd.read_csv(self._filename_underlying, sep=';', index_col=0)
        self._df_asset.index = pd.to_datetime(self._df_asset.index)

        self._df = pd.read_csv(self._filename, sep=";")
        self._dict_df = self._split_price_dates()

        cache_key = (self._filename, self._filename_underlying)
        cached_om = get_from_cache("OptionMarket", cache_key)
        if cached_om is None:
            self._spot = None
            self._options_matrices = self._build_options_matrix()
            cached_om = self
            set_in_cache("OptionMarket", cache_key, cached_om)
        pass

    def _split_price_dates(self):
        """
        Function to split the dataset in DFs for each pricing dates found.
        """
        dfs = {}
        #Get the list of pricing dates
        pricing_dates = self._df["price_date"].unique()
        for date in pricing_dates:
            #Get the dataframe for each pricing date
            df_date = self._df[self._df["price_date"] == date].copy()
            #Get the list of columns to keep
            columns_to_keep = ["price_date", "expiration", "strike", "last", "implied_volatility", "type", "volume"]
            #Keep only the columns to keep
            df_date = df_date[columns_to_keep]
            #Add the dataframe to the dictionary
            dfs[date] = df_date
        return dfs

    def _build_options_matrix(self):
        """
        Function to build the options matrix for each pricing date.
        """
        matrices = {}
        for date in self._dict_df.keys():
            df = self._dict_df[date]
            options_matrix = defaultdict(lambda: {'call': [], 'put': []})    
            
            for _, row in df.iterrows():
                option_type = row['type'].lower()
                if option_type == "call":
                    t = OptionType.CALL
                elif option_type == "put":
                    t = OptionType.PUT
                else:
                    print(f"Invalid Option type: {option_type}")

                option = VanillaOption(row['price_date'], row['expiration'], t, row['strike'], price=row['last'], volume=row['volume'])
                options_matrix[row['expiration']][option_type].append(option)
            
            matrices[date]=options_matrix
        return matrices

    def get_options_for_moneyness(self, price_date:str, maturity:str, moneyness_bounds:tuple=BOUNDS_MONEYNESS, calibrate_on_OTM:bool=OTM_CALIBRATION, calibrate_on_volume:bool=VOLUME_CALIBRATION, volume_bounds:float=VOLUME_THRESHOLD) -> list:
        """
        Function to get options in a certains moneyness for a specified pricing date, and specified maturity.

        Input:
        - spot (float, non optional): spot price of the underlying asset
        - price_date (string, optional): price date of the options
        - maturity (string, optional): maturity date of the options
        - moneyness_bounds (tuple, optional): tuple of min and max moneyness to filter the options (in % or 0.00 format)
        """
        self._spot = self._df_asset.loc[pd.to_datetime(price_date, format="%d/%m/%Y"),"4. close"]
        #Taking correct options
        options_for_dates = self._options_matrices[price_date][maturity]
        options_for_moneyness = []
        k_moneyness = lambda strike:np.log(strike/self._spot)
        if calibrate_on_volume == True:
            count, volume = 0, 0
            for option in options_for_dates['call']:
                count+=1
                volume+=option._volume
            for option in options_for_dates['put']:
                count+=1
                volume+=option._volume
            volume/=count

            if calibrate_on_OTM == True:
                for option in options_for_dates['call']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) >= 0 and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
                for option in options_for_dates['put']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) <= 0 and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
            else:
                for option in options_for_dates['calls']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
                for option in options_for_dates['puts']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and option._volume>volume_bounds*volume:
                        options_for_moneyness.append(option)
        
        else:
            if calibrate_on_OTM == True:
                for option in options_for_dates['call']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) >= 0:
                        options_for_moneyness.append(option)
                for option in options_for_dates['put']:
                    if k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]) and k_moneyness(option._strike) <= 0:
                        options_for_moneyness.append(option)
            else:
                for option in options_for_dates['calls']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]):
                        options_for_moneyness.append(option)
                for option in options_for_dates['puts']:
                    if  k_moneyness(option._strike) >= np.log(moneyness_bounds[0]) and k_moneyness(option._strike) <= np.log(moneyness_bounds[1]):
                        options_for_moneyness.append(option)
        return options_for_moneyness
    
    def get_values_for_calibration_SVI(self, price_date:str, maturity:str, moneyness_bounds:tuple=BOUNDS_MONEYNESS, calibrate_on_OTM:bool=OTM_CALIBRATION, calibrate_on_volume:bool=VOLUME_CALIBRATION, volume_bounds:float=VOLUME_THRESHOLD) -> list:
        """
        Function to get the values for the calibration of the SSI model.

        Input:
        - spot (float, non optional): spot price of the underlying asset
        - price_date (string, optional): price date of the options
        - maturity (string, optional): maturity date of the options
        - moneyness_bounds (tuple, optional): tuple of min and max moneyness to filter the options (in % or 0.00 format)
        """
        options = self.get_options_for_moneyness(price_date, maturity, moneyness_bounds, calibrate_on_OTM, calibrate_on_volume, volume_bounds)
        return [option._type for option in options], [option._strike for option in options], [option._price for option in options], self._spot, options[0].T, options

    def get_values_for_calibration_Heston(self, price_date:str, maturity:str, moneyness_bounds:tuple=BOUNDS_MONEYNESS, calibrate_on_OTM:bool=OTM_CALIBRATION, calibrate_on_volume:bool=VOLUME_CALIBRATION, volume_bounds:float=VOLUME_THRESHOLD) -> list:
        """
        Function to get the values for the calibration of the Heston model.

        Input:
        - spot (float, non optional): spot price of the underlying asset
        - price_date (string, optional): price date of the options
        - maturity (string, optional): maturity date of the options
        - moneyness_bounds (tuple, optional): tuple of min and max moneyness to filter the options (in % or 0.00 format)
        """
        options = self.get_options_for_moneyness(price_date, maturity, moneyness_bounds, calibrate_on_OTM, calibrate_on_volume, volume_bounds)
        return self._spot, options

    def get_spot(self, price_date:str):
        return self._df_asset.loc[pd.to_datetime(price_date, format=FORMAT_DATE),"4. close"]

#Classe de calibration SSVI:
class SSVICalibration:
    """
    Class to calibrate SSVI parameters for a given underlying stock, based on SVI calibration ATM for different maturities.
    Repricing the ATMs options for each maturity, and calibrating the SSVI parameters for ATM options.
    Finally, calibrating the whole SSVI equations to get a full Volatility Surface.
    Be carefull to pass only options for the pricings, need to be cleaned on wanted moneyness levels and liquidity.
    Input:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - data_path: path to the options dataset.
    - file_name_underlying: path of the underlying asset file.
    - pricing_date: date of the desired SSVI.
    - moneyness_level: tuple of min and max moneyness to filter the options (in % or 0.00 format, optional).
    - OTM_calibration: boolean to calibrate on OTM options or not (optional).

    Returns:
    - params: dictionary of SSVI parameters for all maturities.    
    """
    def __init__(self, model:str, data_path:str, file_name_underlying:str, pricing_date:str, moneyness_level:tuple=BOUNDS_MONEYNESS, OTM_calibration:bool=OTM_CALIBRATION, div_rate:float=BASE_DIV_RATE, currency:str=BASE_CURRENCY, rate:float=BASE_RATE, initial_ssvi:list=INITIAL_SSVI, ssvi_method:str=SSVI_METHOD, ssvi_options:dict=OPTIONS_SOLVER_SSVI) -> None:
        self._model = model
        self._model_obj = dict_models[model]
        self._pricing_date = pricing_date

        cache_key = (data_path, file_name_underlying)
        cached_om = get_from_cache("OptionMarket", cache_key)
        if cached_om is None:
            self._option_market = OptionMarket(data_path, file_name_underlying)
            cached_om = self._option_market
            set_in_cache("OptionMarket", cache_key, cached_om)
        else:
            self._option_market = cached_om
        
        self._maturities = list(self._option_market._options_matrices[self._pricing_date].keys())
        self._moneyness_level = moneyness_level
        self._OTM_calibration = OTM_calibration
        self._initial_params_ssvi = initial_ssvi
        self._ssvi_method = ssvi_method
        self._ssvi_options = ssvi_options

        self._div_rate = div_rate
        self._currency = currency
        self._rate = rate
        self._spot = None
        self._options_for_calibration = None
        self._maturities_t = {}
        self._atm_options = {}
        self._ssvi_params = {}

        self._params = self._params_svis
        self._calibration_prices = np.array([option._price for option in self._options_for_calibration])
        self._atm_prices = self._reprice_ATM_options

        self.calibrate_SSVI()
        
        cache_key = (data_path, file_name_underlying)
        cached_ssvi = get_from_cache("SSVICalibration", cache_key)
        if cached_ssvi is None:
            cached_ssvi = self
            set_in_cache("SSVICalibration", cache_key, cached_ssvi)

        cached_params = get_from_cache("SSVI", cache_key)
        if cached_params is None:
            cached_params = self._ssvi_params
            set_in_cache("SSVI", cache_key, cached_params)
    
    @property
    def _params_svis(self)->dict:
        """
        Returns a dict of SVI parameters for all the maturities for a given dict of options (equal one maturity).
        """
        params = {}
        res_options_for_calibration = []
        for maturity_date in self._maturities:
            list_types, list_strikes, list_prices, self._spot, t_options, options_for_calibration = self._option_market.get_values_for_calibration_SVI(self._pricing_date, maturity_date, self._moneyness_level , self._OTM_calibration)
            if list_types is not None:
                pricer = OptionPricer(self._pricing_date, maturity_date, model=self._model, spot=self._spot, div_rate=self._div_rate, currency=self._currency, rate=self._rate, notional=1)
                params[maturity_date]= pricer.svi_params(list_types, list_strikes, list_prices)
                res_options_for_calibration.extend(options_for_calibration)
                self._maturities_t[maturity_date]=t_options
        
        self._options_for_calibration = res_options_for_calibration
        return params

    @property
    def _reprice_ATM_options(self)->dict:
        """
        Use the params to price ATM implied volatility for each maturity and use those to calibrate phi function of SSVI.
        """
        prices = {}
        for maturity_date in self._maturities:
            params = self._params[maturity_date]
            if params is not None: #np.log(1) = 0
                atm_vol = np.sqrt((params[0] + params[1] * (params[2]*(-params[3]) + np.sqrt((-params[3])**2 + params[4]**2)))/self._maturities_t[maturity_date])
                pricer = OptionPricer(start_date=self._pricing_date, end_date=maturity_date, type=OptionType.CALL, model=self._model, spot=self._spot, strike=self._spot, div_rate=self._div_rate, rate=self._rate, sigma=atm_vol)
                prices[self._maturities_t[maturity_date]] = pricer.price
                self._atm_options[maturity_date] = pricer.get_option()
        return prices

    def _calibrate_theta(self)->list:
        """
        Used to calibrate phi function of the SSVI.
        """
        fct_theta=lambda k,v_o,v_inf,t: (((1-np.exp(-k*t))/(k*t))*(v_o-v_inf)+v_inf)*t
        fct_valo=lambda option, theta: self._model_obj(option, theta)

        def new_initial(bounds):
            perturbation = np.ones(len(self._initial_params_ssvi[0:3])) * np.random.uniform(-5,5)
            new_init = self._initial_params_ssvi[0:3]*perturbation
            for i in range(len(bounds)):
                if bounds[i][1] == None:
                    bounds[i][1] = 1000000
                if bounds[i][0] == None:    
                    bounds[i][0] = -1000000

            if all(new_init[i] < bounds[i][1] and new_init[i] > bounds[i][0] for i in range(len(self._initial_params_ssvi[0:3]))):
                return new_init
            else:
                return new_initial(bounds)

        def objective(params)->float:
            """
            Objective function to minimize.
            """
            k, v_o, v_inf = params
            maturities = np.array(list(self._maturities_t.values()))
            theta_vec = np.sqrt(fct_theta(k,v_o,v_inf, maturities)/maturities)
            prices = [fct_valo(option, theta).price(self._spot) for option, theta in zip(list(self._atm_options.values()), theta_vec)]
            return np.sum((np.array(prices)-np.array(list(self._atm_prices.values())))**2)
        
        bounds = ([None,None], [1e-6,None], [1e-6,None]) #((1e-16,10), (1e-6,5), (1e-6,5))
        result = minimize(objective, self._initial_params_ssvi[0:3], bounds=bounds, method=self._ssvi_method, options=self._ssvi_options)

        if result.success and result["fun"]<20:
            return result.x
        elif result.success and result["fun"]>20:
            self._initial_params_ssvi[0:3]=new_initial(bounds)
            return self._calibrate_theta()
        else:
            print(f"Optimization failed.") #Method: {self._method}, Tolerance: {self._tolerance}, Max Iterations: {count}, Bounds: {self._bounds}, Starting Point: {self._starting_point}")
            return None
        
    def calibrate_SSVI(self):
        """
        Calibrate the SSVI parameters for all maturities.
        """
        k, v_0, v_inf = self._calibrate_theta()
        fct_theta=lambda k,v_o,v_inf,t: (((1-np.exp(-k*t))/(k*t))*(v_o-v_inf)+v_inf)*t
        fct_prices=lambda option, vol: self._model_obj(option,vol).price(self._spot)

        def ssvi_total_variance(thetas, k_vec, rho, mu, l):
            """
            SSVI total variance function.
            """
            phi = mu*(thetas **(-l))
            term = phi * k_vec + rho
            sqrt_term = np.sqrt(term**2 + (1 - rho**2))
            return 0.5 * thetas * (1 + rho * phi * k_vec + sqrt_term)
        
        vec_maturities = np.array([option.T for option in self._options_for_calibration])
        vec_thetas = np.array(fct_theta(k,v_0,v_inf, vec_maturities))
        vec_k = np.array([np.log(option._strike/self._spot) for option in self._options_for_calibration])

        def objective(params)->float:
            """
            Objective function to minimize.
            """
            rho, mu, l = params
            vec_w = ssvi_total_variance(vec_thetas, vec_k, rho, mu, l)
            ssvi_vol = np.sqrt(vec_w/vec_maturities)
            prices = [fct_prices(option, vol) for option, vol in zip(self._options_for_calibration, ssvi_vol)]
            return np.sum((np.array(prices)-self._calibration_prices)**2)

        bounds = ((-0.9999,0.9999), (1e-16, None), (1e-16, 1))
        result = minimize(objective, self._initial_params_ssvi[-3:], bounds=bounds, method=self._ssvi_method, options=self._ssvi_options)
        if result.success:
            rho, mu, l = result.x
            self._ssvi_params = {"K": k, "v_0": v_0, "v_inf": v_inf, "rho": rho, "mu": mu, "l": l}
            pass
        else:
            print(f"Calibration failed.")
            pass

    def get_options_calibration(self):
        """
        Returns the options used for calibration.
        """
        return self._options_for_calibration

    def get_spot_SSVI(self):
        return self._spot

    def get_ssvi_params(self):
        return self._ssvi_params
    
    @lru_cache(maxsize=1000)
    def __call__(self, K, t):
        """
        Return implied volatility σ(K, t)
        """
        fct_theta=lambda k,v_o,v_inf,t: (((1-np.exp(-k*t))/(k*t))*(v_o-v_inf)+v_inf)*t

        k = np.log(K/self._spot)
        params = self.get_ssvi_params()
        kappa, v_0, v_inf, rho, mu, l = params['K'], params['v_0'], params['v_inf'], params['rho'], params['mu'], params['l']

        theta = fct_theta(kappa, v_0, v_inf, t)
        phi = mu * theta* l
        term = phi * k + rho
        sqrt_term = np.sqrt(term ** 2 + (1 - rho ** 2))
        w = 0.5 * theta * (1 + rho * phi * k + sqrt_term)
        sigma = np.sqrt(w / t)
        return sigma

#Classe LocalVol:
class DupireLocalVol:
    """
    Class for local volatility parametrization, used to calibrate / fit local volatility surface.
    """
    def __init__(self, model:str, data_path:str, file_name_underlying:str, pricing_date:str, moneyness_level:tuple=BOUNDS_MONEYNESS, OTM_calibration:bool=OTM_CALIBRATION, div_rate:float=BASE_DIV_RATE, currency:str=BASE_CURRENCY, rate:float=BASE_RATE, delta_k:float=BASE_DELTA_K, limits_K:tuple=BASE_LIMITS_K) -> None:
        cache_key = (data_path, file_name_underlying)

        cached_dupire = get_from_cache("DupireLocalVol", cache_key)
        if cached_dupire is not None:
            self._params = cached_dupire._params
            self._implied_vol_df = cached_dupire._implied_vol_df
            self._strikes_supported = cached_dupire._strikes_supported
            self._maturities_t = cached_dupire._maturities_t
            self._options_for_calibration = cached_dupire._options_for_calibration
        else:
            self._params = None
            self._implied_vol_df = None
            self._strikes_supported = []
            self._maturities_t = {}
            self._options_for_calibration = None
        
        self._model = model
        self._model_obj = dict_models[model]
        self._pricing_date = pricing_date
        self._moneyness_level = moneyness_level
        self._OTM_calibration = OTM_calibration
        self._div_rate = div_rate
        self._currency = currency
        self._rate = rate
        self._delta_K = delta_k
        self._limits_K = limits_K
        
        cache_key = (data_path, file_name_underlying)
        self._option_market = get_from_cache("OptionMarket", cache_key)
        if self._option_market is None:
            self._option_market = OptionMarket(data_path, file_name_underlying)
            set_in_cache("OptionMarket", cache_key, self._option_market)

        self._maturities = list(self._option_market._options_matrices[self._pricing_date].keys())[:-1]
        self._spot = self._option_market.get_spot(self._pricing_date)

        if cached_dupire is None:
            self._params = self._params_svis
            set_in_cache("SVI_PARAMS", cache_key, self._params)

            self._implied_vol_df = self._build_implied_vol_matrix()
            set_in_cache("Vol_matrix", cache_key, self._implied_vol_df)

            set_in_cache("DupireLocalVol", cache_key, self)
        pass

    @property
    def _params_svis(self)->dict:
        """
        Returns a dict of SVI parameters for all the maturities for a given dict of options (equal one maturity).
        """
        params = {}
        res_options_for_calibration = []
        for maturity_date in self._maturities:
            list_types, list_strikes, list_prices, self._spot, t_options, options_for_calibration = self._option_market.get_values_for_calibration_SVI(self._pricing_date, maturity_date, self._moneyness_level , self._OTM_calibration)
            if list_types is not None:
                pricer = OptionPricer(self._pricing_date, maturity_date, model=self._model, spot=self._spot, div_rate=self._div_rate, currency=self._currency, rate=self._rate, notional=1)
                params[maturity_date]= pricer.svi_params(list_types, list_strikes, list_prices)
                res_options_for_calibration.extend(options_for_calibration)
                self._maturities_t[maturity_date]=t_options
        
        self._options_for_calibration = res_options_for_calibration
        return params

    def _build_implied_vol_matrix(self) -> pd.DataFrame:
        """
        With the SVIs parameters, rebuild an implied vol matrix, complete that will serve as a basis for Dupuire formula.
        The idea is that we already have the maturities. And for each maturities, and a given delta_k, we will get strikes in some limits.
        Then, get implied vol for each K and maturities.
        """
        fct_vi=lambda a,b,p,m,s,k: a + b * (p * (k - m) + np.sqrt((k - m)**2 + s**2))
        iv_df = pd.DataFrame()

        K = self._spot
        vector_K = []
        while K >= self._spot * self._limits_K[0]:
            K -= self._spot * self._delta_K
            vector_K.append(K)
        K = self._spot
        while K <= self._spot * self._limits_K[1]:
            K += self._spot * self._delta_K
            vector_K.append(K)
        vector_K=np.array(sorted(vector_K))

        for t in range(len(self._maturities)):
            a,b,p,m,s = self._params[self._maturities[t]]
            k_vec = np.array(np.log(vector_K/self._spot))
            w_vec = fct_vi(a,b,p,m,s,k_vec)
            vols = np.sqrt(w_vec/self._maturities_t[self._maturities[t]])
            row = pd.Series(data=vols, index=vector_K, name=self._maturities_t[self._maturities[t]])
            iv_df = pd.concat([iv_df, row.to_frame().T], axis=0)

        self._strikes_supported=list(iv_df.columns)
        return iv_df

    def get_implied_vol_matrix(self):
        return self._implied_vol_df

    @lru_cache(maxsize=1000)
    def get_local_implied_vol(self, maturity:float, strike:float) -> float:
        """
        Get the local implied vol for a given maturity and strike, from Dupire Formula.
        """
        fct_w = lambda a,b,p,m,s,k: a + b * (p * (k - m) + np.sqrt((k - m)**2 + s**2))
        fct_d_sigma_T = lambda sigma_K_T1, sigma_K_T2, T1, T2: (sigma_K_T2 - sigma_K_T1)/(T2 - T1)
        fct_d_sigma_K = lambda sigma_K1_T, sigma_K2_T, K1, K2: (sigma_K2_T - sigma_K1_T)/(K2 - K1)

        def fct_d_sigma2_K(K1, K, K2, sigma_K1_T, sigma_K_T, sigma_K2_T):
            h1 = K - K1
            h2 = K2 - K
            return (2 / (h1 + h2)) * ((sigma_K2_T - sigma_K_T) / h2 - (sigma_K_T - sigma_K1_T) / h1)

        k = np.log(strike/self._spot)
        under_mat, upper_mat = self.get_closest_maturities(maturity)
        under_strike, upper_strike = self.get_closest_strikes(strike)
        
        dates = np.array(list(self._maturities_t.keys()))
        t = np.array(list(self._maturities_t.values()))
        date_params_T2 = dates[np.where(t==upper_mat)[0]][0]
        params_T2 = self._params[date_params_T2]

        s_IMP = np.sqrt(fct_w(*params_T2, k)/upper_mat)
        s2_IMP=s_IMP**2

        date_params_T1 = dates[np.where(t==under_mat)[0]][0]
        params_T1 = self._params[date_params_T1]
        sigma_K_T1=np.sqrt(fct_w(*params_T1, k)/under_mat)

        sigma_K1_T = self._implied_vol_df.loc[upper_mat, under_strike]
        sigma_K2_T = self._implied_vol_df.loc[upper_mat, upper_strike]

        d_sigma_T = fct_d_sigma_T(sigma_K_T1, s_IMP, under_mat, upper_mat)
        d_sigma_K = fct_d_sigma_K(sigma_K1_T, sigma_K2_T, under_strike, upper_strike)
        d_sigma2_K = fct_d_sigma2_K(under_strike, strike, upper_strike, sigma_K1_T, s_IMP, sigma_K2_T)

        numerator=s2_IMP + 2 * s_IMP * maturity * (d_sigma_T + (self._rate - self._div_rate) * strike * d_sigma_K)
        denominator=((1+strike*d_sigma_K*np.sqrt(maturity))**2) + (s_IMP*(strike**2)*maturity)*(d_sigma2_K-self._div_rate*(d_sigma_K**2)*np.sqrt(maturity))

        res = np.sqrt(numerator/denominator)
        if np.isnan(res):
            res = sigma_K_T1
        return res

    def get_closest_maturities(self, maturity:float)->list:
        """
        Get the 2 closest maturities for a given maturity.
        """
        maturities = list(self._maturities_t.values())
        maturities.sort()
        under_mat, upper_mat = maturities[0], maturities[-1]
        for i in range(len(maturities)):
            if maturities[i]>under_mat and maturities[i]<=maturity:
                under_mat = maturities[i]
            if maturities[i]<upper_mat and maturities[i]>=maturity:
                upper_mat = maturities[i]
        return under_mat, upper_mat
    
    def get_closest_strikes(self, strike: float):
        """
        Return the closest strike below and above the given strike.
        Assumes self._implied_vol_df.columns is sorted.
        """
        strikes = sorted(list(self._implied_vol_df.columns))
        
        under = max([k for k in strikes if k < strike], default=strikes[0])
        over = min([k for k in strikes if k > strike], default=strikes[-1])

        return under, over

#Classe helper pour préparer/calibrer l'Heston model:
class HestonHelper:
    """
    Class to calibrate Heston model parameters, there is the choice between calibrating on SSVI or actual market prices.
    
    Input:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - data_path: path to the options dataset.
    - file_name_underlying: path of the underlying asset file.
    - pricing_date: starting date.
    - moneyness_level: tuple of min and max moneyness to filter the options (in % or 0.00 format, optional).
    - OTM_calibration: boolean to calibrate on OTM options or not (optional).
    - div_rate: dividend rate (optional).
    - currency: currency of the underlying asset (optional).
    - rate: risk-free rate (optional).
    - K_delta: delta for the strikes (optional).
    - max_t: maximum maturity for the options (optional).
    - t_interval: time interval for the options (optional).
    - initial_params_heston: initial parameters for the Heston model (optional).
    - heston_method: method used for optimization (optional).
    - bounds_heston: bounds for the Heston model parameters (optional).
    - options_heston: options for the optimization (optional).
    """
    def __init__(self,model:str,data_path:str,file_name_underlying:str,pricing_date:str,moneyness_level:tuple=BASE_LIMITS_K_H,\
                K_delta:float=BASE_DELTA_K, OTM_calibration:bool=OTM_CALIBRATION, div_rate:float=BASE_DIV_RATE,\
                currency:str=BASE_CURRENCY, rate:float=BASE_RATE, type_calibration:str=BASE_CALIBRATION_HESTON, max_t:float=BASE_MAX_T, t_interval:float=BASE_T_INTERVAL,\
                initial_params_heston:list=INITIAL_HESTON, heston_method:str=HESTON_METHOD, bounds_heston:tuple=HESTON_BOUNDS, options_heston:dict=HESTON_CALIBRATION_OPTIONS)->None:
        
        self._type_calibration=type_calibration
        self._model=model
        self._data_path=data_path
        self._file_name_underlying=file_name_underlying
        self._pricing_date=pricing_date
        self._moneyness_level=moneyness_level
        self._OTM_calibration=OTM_calibration
        self._div_rate=div_rate
        self._rate=rate
        self._currency=currency

        self._max_T=max_t
        self._time_interval=t_interval
        self._K_delta=K_delta
    
        self._initial_params_heston=initial_params_heston
        self._heston_method=heston_method
        self._bounds_heston=bounds_heston
        self._options_heston=options_heston

        self._results_heston=None

        cache_key = (data_path, file_name_underlying)
        cached_params = get_from_cache("HestonHelper", cache_key)
        if cached_params is None:
            if self._type_calibration=="SSVI":
                cache_key = (data_path, file_name_underlying)
                cached_calibrator = get_from_cache("SSVICalibrator", cache_key)
                if cached_calibrator is None:
                    ssvi_calibrator=SSVICalibration(model=self._model, data_path=self._data_path, file_name_underlying=self._file_name_underlying, pricing_date=self._pricing_date, moneyness_level=self._moneyness_level, OTM_calibration=self._OTM_calibration, div_rate=self._div_rate, currency=self._currency, rate=self._rate)
                else:
                    ssvi_calibrator = cached_calibrator

                self._spot=ssvi_calibrator.get_spot_SSVI()
                self._options_for_calibration=ssvi_calibrator._options_for_calibration
                self._options_for_calibration=self._options_for_calibration[:int(len(self._options_for_calibration)*CUTOFF_H)]
                self._ssvi_function=ssvi_calibrator.__call__
                self._results_heston=self._calibrate_on_ssvi()
            elif self._type_calibration=="Market":
                options = []
                
                cache_key = (data_path, file_name_underlying)
                cached_om = get_from_cache("OptionMarket", cache_key)
                if cached_om is None:
                    option_market=OptionMarket(self._data_path, self._file_name_underlying)
                else:
                    option_market = cached_om
                
                for maturity in list(option_market._options_matrices[self._pricing_date].keys()):
                    self._spot, opt=option_market.get_values_for_calibration_Heston(self._pricing_date, maturity, self._moneyness_level, self._OTM_calibration)
                    options.extend(opt)
                self._options_for_calibration=options
                self._results_heston=self._calibrate_on_market()
            else:
                print("Calibration type not recognized ! Please choose: Market or SSVI")
                pass
            set_in_cache(HestonHelper, cache_key, self._results_heston)
        else:
            self._results_heston = cached_params
        
        pass

    @lru_cache(maxsize=1000)
    def _CFHeston(self, r, tau, kappa, eta, theta, v0, rho):
        """
        Heston characteristic function.

        Inputs:
        - r: Risk-free rate
        - tau: Time to maturity
        - kappa: Mean reversion speed
        - eta: Volatility of volatility
        - theta: Long-term volatility
        - v0: Initial volatility
        - rho: Correlation between asset and volatility
        """
        i = complex(0,1)
        def cf(u):
            d = np.sqrt((kappa - eta * rho * i * u)**2 + eta**2 * (u**2 + i * u))
            g = (kappa - eta * rho * i * u - d) / (kappa - eta * rho * i * u + d)
            exp_dt = np.exp(-d * tau)
            
            A = i * u * r * tau + \
                (kappa * theta / eta**2) * ((kappa - eta * rho * i * u - d) * tau - 2 * np.log((1 - g * exp_dt) / (1 - g)))
            C = ((kappa - eta * rho * i * u - d) / eta**2) * ((1 - exp_dt) / (1 - g * exp_dt))

            return np.exp(A + C * v0)
        return cf

    def _calibrate_on_ssvi(self)->dict:
        """
        Calibration of the Heston model's parameters on the SSVI calibrated on the market. 
        """
        market_prices=[]
        market_strikes=[]
        market_T=[]
        for option in self._options_for_calibration:
            sigma = self._ssvi_function(option._strike, option.T)
            option_pricer=OptionPricer(self._pricing_date, option._end_date, OptionType.CALL, self._model, self._spot, option._strike, self._div_rate, sigma=sigma, rate=self._rate)
            market_prices.append(option_pricer.price)
            market_strikes.append(option._strike)
            market_T.append(option.T)
        return self._calibrate_Heston(market_prices, market_strikes, market_T)
        
    def _calibrate_on_market(self)->dict:
        """
        Calibration of the Heston model's parameters on the market data directly.
        """
        market_prices=[]
        market_strikes=[]
        market_T=[]
        for option in self._options_for_calibration:
            market_prices.append(option._price)
            market_strikes.append(option._strike)
            market_T.append(option.T)
        return self._calibrate_Heston(market_prices, market_strikes, market_T)

    def _calibrate_Heston(self, market_prices:list, market_strikes:list, market_T:list)->dict:
        market_strikes=np.array(market_strikes)
        market_T=np.array(market_T)

        def error(params, P, K, T):
            v0, kappa, theta, eta, rho = params
            cf = self._CFHeston(self._rate, T, kappa, eta, theta, v0, rho)
            try:
                p_fourrier=self._heston_call_price_fourrier(K, T, cf)
                return (p_fourrier-P)**2
            except Exception:
                return 1e6

        def objective(params)->float:
            errors=Parallel(n_jobs=N_CORES)(delayed(error)(params, P, K, T) for P, K, T in zip(market_prices, market_strikes, market_T))
            return sum(errors)

        result = minimize(objective, self._initial_params_heston, method=self._heston_method, bounds=self._bounds_heston, options=self._options_heston)
        if result.success:
            return dict(zip(['v0', 'kappa', 'theta', 'eta', 'rho'], result.x))
        else:
            raise ValueError('Cannot reach Heston Model parametrization')

    def _heston_call_price_fourrier(self, K, T, cf):
        i = complex(0.0,1.0)
        def integrand_pj(u,j):
            if j == 1:
                numerator = cf(u-i)
                denominator = cf(-i)
                integrand = np.real(np.exp(-i * u * np.log(K)) * numerator / (i*u*denominator))
            elif j == 2:
                integrand = np.real(np.exp(-i*u*np.log(K)) * cf(u)/(i*u))
            return integrand

        P1 = 0.5 + (1/np.pi) * quad(lambda u: integrand_pj(u, 1), 0, 50, limit=50)[0]
        P2 = 0.5 + (1/np.pi) * quad(lambda u: integrand_pj(u, 2), 0, 50, limit=50)[0]
        return self._spot * P1 - np.exp(-self._rate*T) * K * P2

    def get_heston_params(self)->dict:
        return self._results_heston

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les différentes classes pricers--------------------
#-------------------------------------------------------------------------------------------------------
#Options pricer: European style options
class OptionPricer:
    """
    Class for European option pricing (usable for Barriers / vanilla european options)
    """
    def __init__(self, start_date:str, end_date:str, type:str=OptionType.CALL, barrier_type:str=None, model:str=BASE_MODEL,
                 spot:float = None, strike: float = BASE_STRIKE, barrier_strike:float=None, div_rate: float = BASE_DIV_RATE,
                 day_count: str = CONVENTION_DAY_COUNT, rolling_conv: str = ROLLING_CONVENTION,
                 notional: float = 1, format_date: str = FORMAT_DATE, currency: str = BASE_CURRENCY,
                 sigma: float = None, rate: float = BASE_RATE, price: float = None,
                 model_parameters: dict = None, nb_paths: float = NUMBER_PATHS_H, nb_steps: float = NB_STEPS_H,
                 data_path: str = FILE_PATH, file_name_underlying: str = FILE_UNDERLYING) -> None:

        self._data_path = data_path
        self._file_name_underlying = file_name_underlying
        
        self._model_name = model
        self._start_date = start_date
        self._end_date = end_date
        self._type = type
        self._strike = strike
        self._div_rate = div_rate
        self._day_count = day_count
        self._rolling_conv = rolling_conv
        self._notional = notional
        self._format_date = format_date
        self._currency = currency
        self._sigma = sigma
        self._rate = rate
        self._barrier_type = barrier_type
        self._barrier_strike = barrier_strike

        if spot is None:
            data = pd.read_csv(self._file_name_underlying, sep=';', index_col=0)
            data.index = pd.to_datetime(data.index)
            spot = data.loc[pd.to_datetime(self._start_date, format=FORMAT_DATE),"4. close"]
        self._spot = spot

        if model_parameters is not None:
            self._model_parameters = model_parameters
        elif self._model_name == "Heston":
            self._model_parameters = get_heston_params_from_csv(self._start_date)
        else:
            self._model_parameters = None

        self._nb_paths = nb_paths
        self._nb_steps = nb_steps
        self._spots_paths = None
        self._payoff = None
        self._price = price

        if self._type == OptionType.CALL or self._type == OptionType.PUT:
            self._option = VanillaOption(start_date=start_date, end_date=end_date, type=type, strike=strike,
                                     notional=notional, currency=currency, div_rate=div_rate)
            
        elif self._type == BarrierType.CALL_DOWN_IN or self._type == BarrierType.CALL_DOWN_OUT or self._type == BarrierType.CALL_UP_IN or self._type == BarrierType.CALL_UP_OUT or self._type == BarrierType.PUT_DOWN_IN or self._type == BarrierType.PUT_DOWN_OUT or self._type == BarrierType.PUT_UP_IN or self._type == BarrierType.PUT_UP_OUT:
            self._option = BarrierOption(start_date=start_date, end_date=end_date, type=type, strike=strike,
                                     notional=notional, currency=currency, div_rate=div_rate,
                                     barrier_strike=barrier_strike)
        else:
            ValueError(f"Option Type {self._type} not recognized !")

        self._local_vol_model = None
        if self._model_name == "Dupire":
            cache_key = (data_path, file_name_underlying)
            cached_lv = get_from_cache("DupireLocalVol", cache_key)
            if cached_lv is None:
                cached_lv = DupireLocalVol(BASE_MODEL, self._data_path, self._file_name_underlying, self._start_date, BOUNDS_MONEYNESS, OTM_CALIBRATION, self._div_rate, self._currency, self._rate)
                set_in_cache("DupireLocalVol", cache_key, cached_lv)
            self._local_vol_model = self._local_vol_model = cached_lv

        self._model = self._build_model()

    @property
    def price(self)->float:
        if self._model_name == "Black-Scholes-Merton":
            price = self._model.price(self._spot)
            self._payoff = price * np.exp(self._rate * self._option.T) * self._notional
            return price
        else:
            price = self._model.price(self._spot)
            self._payoff, self._spots_paths = self._model._payoffs, self._model._spots
            return price

    @property
    def delta(self)->float:
        return self._greek("delta")

    @property
    def gamma(self)->float:
        return self._greek("gamma")

    @property
    def vega(self)->float:
        return self._greek("vega")

    @property
    def theta(self)->float:
        return self._greek("theta")

    @property
    def rho(self)->float:
        return self._greek("rho")

    def _build_model(self):
        if self._model_name == "Black-Scholes-Merton":
            sigma = self._sigma if self._sigma is not None else BASE_SIGMA
            return BSM(self._option, sigma)

        elif self._model_name == "Heston":
            return Heston(self._option, self._model_parameters, self._nb_paths, self._nb_steps)

        elif self._model_name == "Dupire":
            return Dupire(self._option, self._local_vol_model, self._nb_paths, self._nb_steps)

        else:
            raise ValueError(f"Model {self._model_name} not recognized!")

    def _update_model(self):
        self._model = self._build_model()

    def _greek(self, greek_name:str):
        self._update_model()
        return getattr(self._model, greek_name)(self._spot)

    def implied_vol(self, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> float:
        if self._price is None:
            self._price = self.price
        self._model = dict_models["Black-Scholes-Merton"]
        volatility_finder = ImpliedVolatilityFinder(model=self._model, option=self._option, price=self._price, method=method, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point, spot=self._spot)
        return volatility_finder.find_implied_volatility()
    
    def svi_params(self, vector_types:list, vector_strikes:list, vector_prices:list, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> tuple:
        options=[]
        for i in range(len(vector_strikes)):
            options.append(VanillaOption(start_date=self._start_date, end_date=self._end_date, type=vector_types[i], strike=vector_strikes[i], notional=self._notional, currency=self._currency, div_rate=self._div_rate))
        self._model = dict_models["Black-Scholes-Merton"]
        svi_params_finder = SVIParamsFinder(model=self._model, vector_options=options, vector_prices=vector_prices, method_implied_vol=method, spot=self._spot, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point)
        result = svi_params_finder.find_svi_parameters()
        return result

    def get_option(self):
        return self._option

    def __deep_copy__(self):
        return OptionPricer(self._start_date, self._end_date, type=self._type, barrier_type=self._barrier_type, model=self._model_name, spot=self._spot, strike=self._strike, barrier_strike=self._barrier_strike, div_rate=self._div_rate, day_count=self._day_count, rolling_conv=self._rolling_conv, notional=self._notional, format_date=self._format_date, currency=self._currency, sigma=self._sigma, rate=self._rate, model_parameters=self._model_parameters, nb_paths=self._nb_paths, nb_steps=self._nb_steps, price=self._price)

#Autocall:
class AutocallPricer:
    """
    Autocal pricing engine.
    """
    def __init__(self, start_date:str, end_date:str, type:str=AutocallsType.AUTOCALL, model:str=BASE_MODEL_AUTOCALLS,
                 spot:float = None, strike: float = BASE_STRIKE, final_strike:float = None, coupon:float=None, coupon_strike:float=None,
                 protection:float=None, memory:bool=True, exercise_type:float=None, frequency:str=None, div_rate: float = BASE_DIV_RATE, 
                 day_count: str = CONVENTION_DAY_COUNT, rolling_conv: str = ROLLING_CONVENTION,
                 notional: float = BASE_NOTIONAL, format_date: str = FORMAT_DATE, currency: str = BASE_CURRENCY,
                 sigma: float = None, rate: float = BASE_RATE, model_parameters: dict = None, nb_paths: float = NUMBER_PATHS_H, nb_steps: float = NB_STEPS_H,
                 price: float = None, data_path: str = FILE_PATH, file_name_underlying: str = FILE_UNDERLYING) -> None:
        
        self._data_path = data_path
        self._file_name_underlying = file_name_underlying
        
        self._model_name = model
        self._start_date = start_date
        self._end_date = end_date
        self._frequency = frequency
        self._type = type
        self._strike = strike
        self._final_strike = final_strike
        self._coupon=coupon
        self._coupon_strike=coupon_strike
        self._memory=memory
        self._protection_capital=protection
        self._exercise_type=exercise_type

        self._div_rate = div_rate
        self._day_count = day_count
        self._rolling_conv = rolling_conv
        self._notional = notional
        self._format_date = format_date
        self._currency = currency
        self._sigma = sigma
        self._rate = rate

        if spot is None:
            data = pd.read_csv(self._file_name_underlying, sep=';', index_col=0)
            data.index = pd.to_datetime(data.index)
            spot = data.loc[pd.to_datetime(self._start_date, format=FORMAT_DATE),"4. close"]
        self._spot = spot

        if model_parameters is not None:
            self._model_parameters = model_parameters
        elif self._model_name == "Heston":
            self._model_parameters = get_heston_params_from_csv(self._start_date)
        else:
            self._model_parameters = None

        self._nb_paths = nb_paths
        self._nb_steps = nb_steps
        self._spots_paths = None
        self._payoff = None
        self._price = price

        self._option = Autocalls(start_date=self._start_date, end_date=self._end_date, type=self._type,\
                                     strike=self._strike, final_strike=self._final_strike, coupon=self._coupon, coupon_strike=self._coupon_strike, \
                                     protection=self._protection_capital, memory=self._memory, type_opt=self._exercise_type, frequency=self._frequency, notional=self._notional)

        self._local_vol_model = None
        if self._model_name == "Dupire":
            cache_key = (data_path, file_name_underlying)
            cached_lv = get_from_cache("DupireLocalVol", cache_key)
            if cached_lv is None:
                cached_lv = DupireLocalVol(BASE_MODEL, self._data_path, self._file_name_underlying, self._start_date, BOUNDS_MONEYNESS, OTM_CALIBRATION, self._div_rate, self._currency, self._rate)
                set_in_cache("DupireLocalVol", cache_key, cached_lv)
            self._local_vol_model = cached_lv

        self._model = self._build_model()
        pass

    @property
    def price(self)->float:
        if self._model_name == "Black-Scholes-Merton":
            ValueError(f"Model {self._model_name} not supported for autocalls !")
        else:
            paths = self._model._generate_paths(self._spot)
            self._option.set_paths(paths)
            npv, payoff, par_cpn, call_prob = self._option.price()
            return npv, payoff, par_cpn, call_prob

    def _build_model(self):
        if self._model_name == "Black-Scholes-Merton":
            sigma = self._sigma if self._sigma is not None else BASE_SIGMA
            return BSM(self._option, sigma)

        elif self._model_name == "Heston":
            return Heston(self._option, self._model_parameters, self._nb_paths, self._nb_steps)

        elif self._model_name == "Dupire":
            return Dupire(self._option, self._local_vol_model, self._nb_paths, self._nb_steps)

        else:
            raise ValueError("Model not recognized")

    def _update_model(self):
        self._model = self._build_model()

    def implied_vol(self, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> float:
        if self._price is None:
            self._price = self.price
        self._model = dict_models["Black-Scholes-Merton"]
        volatility_finder = ImpliedVolatilityFinder(model=self._model, option=self._option, price=self._price, method=method, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point, spot=self._spot)
        return volatility_finder.find_implied_volatility()
    
    def svi_params(self, vector_types:list, vector_strikes:list, vector_prices:list, method:str=BASE_METHOD_VOL, tolerance:float=TOLERANCE, max_iter:float=MAX_ITER, bounds:tuple=BOUNDS, starting_point:float=STARTING_POINT) -> tuple:
        options=[]
        for i in range(len(vector_strikes)):
            options.append(VanillaOption(start_date=self._start_date, end_date=self._end_date, type=vector_types[i], strike=vector_strikes[i], notional=self._notional, currency=self._currency, div_rate=self._div_rate))
        self._model = dict_models["Black-Scholes-Merton"]
        svi_params_finder = SVIParamsFinder(model=self._model, vector_options=options, vector_prices=vector_prices, method_implied_vol=method, spot=self._spot, tolerance=tolerance, nb_iter=max_iter, bounds=bounds, starting_point=starting_point)
        result = svi_params_finder.find_svi_parameters()
        return result

    def get_option(self):
        return self._option

#Option portolio for option strategies:
class Portfolio:
    """
    Class to regroup and launch pricing of multiple options / products.
    Look at it like an interface with the front-end.
    """
    def __init__(self):
        self._portfolio = {}
        pass

    def _add_product(self, type_product:str, start_date:str, end_date:str, quantity:float=1, strike:float=BASE_STRIKE, barrier_strike:float=None, model:str=BASE_MODEL, spot:float=None, div_rate:float=BASE_DIV_RATE, rate:float=BASE_RATE, day_count:str=CONVENTION_DAY_COUNT, rolling_conv:str=ROLLING_CONVENTION, notional:float=BASE_NOTIONAL, format_date:str=FORMAT_DATE, currency:str=BASE_CURRENCY, sigma:float=None, heston_parameters:dict=None):
        """
        Add a product to the portfolio.
        """
        if type_product not in DICT_PRODUCT or model not in dict_models:
            raise ValueError(f"Product {type_product} or Model {model} not recognized.")
        else:
            key = (type_product, model, strike, start_date, end_date, barrier_strike, div_rate, rate, day_count, rolling_conv, notional, format_date, currency, sigma)
            if key not in self._portfolio:
                if strike<0.1:
                    strike_effective=0.1
                else: 
                    strike_effective = strike
                option_pricer = OptionPricer(start_date, end_date, type=DICT_PRODUCT[type_product], barrier_strike=barrier_strike, model=model, spot=spot, strike=strike_effective, div_rate=div_rate, rate=rate, day_count=day_count, rolling_conv=rolling_conv, notional=notional, format_date=format_date, currency=currency, sigma=sigma, model_parameters=heston_parameters)
                self._portfolio[key] = {'pricer': option_pricer, 'quantity': quantity}
            else:
                self._portfolio[key]['quantity'] += quantity
        print("Product successfully added to portfolio!")
        pass

    def clear_portfolio(self):
        self._portfolio = {}
        pass

    def price_portfolio(self):
        npvs=[]
        pay_offs=[]
        spots=[]

        for key, item in self._portfolio.items():
            pricer = item['pricer']
            price = pricer.price*item['quantity']
            npvs.append(price)
            pay_offs.extend([x * item['quantity'] for x in pricer._payoff])
            spots.extend(pricer._spots_paths)
        return sum(npvs), npvs, pay_offs, spots
    
    def __deep_copy__(self):
        return Portfolio(self._portfolio)