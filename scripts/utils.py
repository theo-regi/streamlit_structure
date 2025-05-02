from constants import CONVENTION_DAY_COUNT, TYPE_INTERPOL, INITIAL_RATE, IMPLIED_VOL_METHODS, SOLVER_METHOD,\
                      SVI_SOLVER_METHOD, INITIAL_SVI, OPTIONS_SOLVER_SVI, FILE_PATH, FILE_UNDERLYING, DATA_PATH,\
                      FORMAT_DATE, BASE_CURRENCY
from datetime import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import holidays
import pandas as pd
import numpy as np
from scipy.optimize import fsolve, minimize
from sklearn.linear_model import LinearRegression
import os
from scripts.functions import optimize_nelson_siegel,nelson_siegel

#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les classes utilitaires----------------------------
#-------------------------------------------------------------------------------------------------------
#Maturity handler :: convention, format, rolling convention, market -> year fraction
class Maturity_handler:
    """
    Classe pour gérer les différents problèmes de maturités -> renvoie une year_fraction
    
    Input: 
        - une convention de day count (30/360, etc) en string
        - un format de date (%d/%m/%Y = 01/01/2025)
        - une rolling convention = comment on gère les jours de marchés fermés (Modified Following, etc)
        - un marché (UE = XECB, US = XNYS, Brésil/Lattam = BVMF, UK = IFEU), (ce n'est pas optimal car pas sur que les jours fériés soient corrects, il faudrait une fonction bloomberg, mais inaccessible hors de l'université)

    Pour l'utiliser, appeller get_year_fraction, avec en input, une date proche, une date éloignée.
    
    Renvoie un float = nombres d'années, 1.49 = +/- 1 ans et 6 mois (fonction des jours fériés, convention, etc).
    """
    def __init__(self, convention: str, format_date: str, rolling_convention: str, market: str) -> None:
        self.__convention = convention
        self.__format_date = format_date
        self.__rolling_convention = rolling_convention
        self.__market = market
        self.__calendar = self.__get_market_calendar(dt.today()-timedelta(days=100*360), dt.today()+timedelta(days=100*360))
        pass

    def __convention_handler(self, valuation_date, end_date) -> float:
        """Returns the corresponding year_fraction (end_date - valuation_date)
            corresponding to the convention of the handler."""
        d1, m1, y1 = valuation_date.day, valuation_date.month, valuation_date.year
        d2, m2, y2 = end_date.day, end_date.month, end_date.year
        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30
        if self.__convention == "30/360":
            return (360*(y2 - y1) + 30 * (m2 - m1) + (d2 - d1))/360
        elif self.__convention == "Act/360":
            delta_days  = (end_date - valuation_date).days
            return delta_days/360
        elif self.__convention == "Act/365":
            delta_days  = (end_date - valuation_date).days
            return delta_days/365
        elif self.__convention == "Act/Act":
            days_count = 0
            current_date = valuation_date
            while current_date < end_date:
                year_end = dt(current_date.year, 12, 31)
                if year_end > end_date:
                    year_end = end_date
                days_in_year = (dt(current_date.year, 12, 31) - dt(current_date.year, 1, 1)).days + 1
                days_count += (year_end - current_date).days / days_in_year
                current_date = year_end + timedelta(days=1)
            return days_count
        else:
            raise ValueError(f"Entered Convention: {self.__convention} is not handled ! (30/360, Act/360, Act/365, Act/Act)")

    def __get_market_calendar(self, start_date, end_date):
        try:
            return holidays.financial_holidays(market=self.__market, years=(range(start_date.year, end_date.year)))
        except:
            raise ValueError(f"Error calendar: {self.__market} is not supported Choose (XECB, IFEU, XNYS, BVMF)")
        pass
        
    def __get_next_day(self, date):
        while date.weekday() >= 5 or date in self.__calendar:
            date += timedelta(days=1)
        return date
    
    def __get_previous_day(self, date):
        while date.weekday() >= 5 or date in self.__calendar:
            date -= timedelta(days=1)
        return date

    def __apply_rolling_convention(self, date):
        if self.__rolling_convention == "Following":
            return self.__get_next_day(date)
        
        elif self.__rolling_convention == "Modified Following":
            new_date = self.__get_next_day(date)
            if new_date.month != date.month:
                return self.__get_previous_day(date)
            else:
                return new_date
            
        elif self.__rolling_convention == "Preceding":
            return self.__get_previous_day(date)
        
        elif self.__rolling_convention == "Modified Preceding":
            new_date = self.__get_previous_day(date)
            if new_date.month != date.month:
                return self.__get_next_day(date)
            else:
                return new_date
        else:
            raise ValueError(f"Rolling Convention {self.__rolling_convention} is not supported ! Choose: Following, Modified Following, Preceding, Modified Preceding")

    def get_year_fraction(self, valuation_date, end_date) -> float:
        """Takes valuatio_date and end_date as strings, convert to datetime
            :: returns year_fraction (float) depending on the self.__convention"""
        #If dates arrives in the strings
        if type(valuation_date) == str:
            valuation_date = dt.strptime(valuation_date, self.__format_date)
        if type(end_date) == str:
            end_date = dt.strptime(end_date, self.__format_date)

        #We need to get the real "openned days" of the market (calendars) = Modified Following, etc.    
        if valuation_date.weekday()>=5 or valuation_date in self.__calendar:
            valuation_date = self.__apply_rolling_convention(valuation_date)
        if valuation_date.weekday()>=5 or valuation_date in self.__calendar:
            end_date = self.__apply_rolling_convention(end_date)
        return self.__convention_handler(valuation_date, end_date)

#Payment Schedule Handler:: first date of the schedule, last date of the schedule, periodicity of dates between, date format.
class PaymentScheduleHandler:
    """
    Classe pour générer des échéanciers de paiements entre une date de départ et une date de fin
    
    Inputs: 
        - valuation_date: date de départ (exemple: aujourd'hui ou t+2 = convention de marché)
        - end_date: dernière date de l'échéancier = date du dernier paiement
        - periodicity: temps entre deux paiements (monthly, quaterly, semi-annually, annually)
        - date_format: format d'input pour les valuation_date et end_date (exemple: %d/%m/%Y)

    utilisation: créer un payement scheduler avec les inputs, appeller build_schedule avec les conventions utilisées + marché.
    
    Renvoie un tuple d'échéances intermédiaires (ex:(0.5, 1, 1.5, 2, 2.5, 3) pour 3 ans paiement semi-annuel)
        ajusté aux jours de marchés fermés + convention de calculs.
    """
    def __init__(self, valuation_date: str, end_date: str, periodicity: str, date_format:str) -> None:
        self.__valuation_date = valuation_date
        self.__end_date = end_date
        self.__periodicity = periodicity
        self.__date_format = date_format
        pass

    def build_schedule(self, convention: str, rolling_convention: str, market: str) -> tuple:
        """Takes a start_date, end_date, periodicity 
            :: returns a tuple of year_fractions tuple because read only."""
        self.__valuation_date = dt.strptime(self.__valuation_date, self.__date_format)
        self.__end_date = dt.strptime(self.__end_date, self.__date_format)
        list_dates = self.__get_intermediary_dates()

        maturityhandler = Maturity_handler(convention, self.__date_format, rolling_convention, market)

        list_year_fractions = []
        #If we need the "t" corresponding to the first date/start date of the product (t=0), adjust list_dates[1:] to list_dates[0:]
        for date in list_dates[1:]:
            list_year_fractions.append(maturityhandler.get_year_fraction(list_dates[0], date))
        return list(list_year_fractions)

    def __get_intermediary_dates(self) -> list:
        """Build a dates list with all intermediary dates between start and end based on periodicity."""
        """Supported periodicity: monthly, quaterly, semi-annually, annually."""
        list_dates = [self.__valuation_date]
        count_date = self.__valuation_date
       
        while count_date < self.__end_date-relativedelta(months=1):
            if self.__periodicity == "monthly":
                count_date += relativedelta(months=1)
                list_dates.append(count_date)
            elif self.__periodicity == "quarterly":
                count_date += relativedelta(months=3)
                list_dates.append(count_date)
            elif self.__periodicity == "semi-annually":
                count_date += relativedelta(months=6)
                list_dates.append(count_date)
            elif self.__periodicity == "annually":
                count_date += relativedelta(years=1)
                list_dates.append(count_date)
            elif self.__periodicity == "none":
                count_date = self.__end_date
                list_dates.append(count_date)
            else:
                raise ValueError(f"Entered periodicity {self.__periodicity} is not supported. Supported periodicity: monthly, quaterly, semi-annually, annually, none.")
        
        list_dates.append(self.__end_date)
        return list_dates
        
#Classe de rate et courbe de taux
class Rates_curve:
    """
    Classe pour gérer les courbes de taux, interpoler, calculer les taux forward, shift de courbe, etc.

    Inputs:
        - path_rate: path du fichier csv contenant les taux
        - flat_rate: taux fixe à appliquer si besoin (optionnel, utilisé pour les courbes flats)

    Utilisation:
        - get_data_rate: retourne les données de la courbe de taux
        - year_fraction_data: ajoute une colonne Year_fraction à la dataframe des taux
        - attribute_rates_curve: ajoute les maturités des produits à la courbe de taux
        - linear_interpol: interpole les taux de la courbe de taux
        - quadratic_interpol: interpole les taux de la courbe de taux
        - Nelson_Siegel_interpol: interpole les taux de la courbe de taux
        - flat_rate: applique un taux fixe à la courbe de taux
        - forward_rate: calcule les taux forward de la courbe de taux
        - create_product_rate_curve: crée une courbe de taux pour un produit donné
        - shift_curve: shift la courbe de taux

    Renvoie la courbe de taux pour un produit donné.
    """
    def __init__(self, path_rate:str, flat_rate:float= None, method:str=None):
        self.__path_rate = path_rate
        self.__flat_rate = flat_rate
        self.__method = method
        self.__data_rate = pd.read_csv(path_rate,sep=";") #Only CSV reader supported for now. No link to external sources developped (not required + not pratical for student's dev).
        self.curve_rate_product = None
        pass

    def get_data_rate(self):
        """
        Fonction pour renvoyer les données de la courbe de taux (depuis un csv).
        """
        return self.__data_rate

    def year_fraction_data(self,convention: str=CONVENTION_DAY_COUNT) -> pd.DataFrame:
        """
        Fonction pour ajouter une colonne Year_fraction à la dataframe des taux.

        Input:
        - convention: convention de calcul de la year_fraction (30/360, 30/365, etc)
        """
        factor_map = {'D': 1, 'W': 7, 'M': 30, 'Y': convention}
        self.__data_rate['Year_fraction'] = self.__data_rate['Pillar'].str[:-1].astype(float) * self.__data_rate['Pillar'].str[-1].map(factor_map) / convention
        self.__data_rate['Year_fraction'] = self.__data_rate['Year_fraction'].round(6)
        return self.__data_rate

    def attribute_rates_curve(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour ajouter les maturités des produits à la courbe de taux.

        Input:
        - product_year_fraction: liste des maturités des produits
        """
        df= pd.DataFrame({"Year_fraction": product_year_fraction})
        df["Year_fraction"]=df["Year_fraction"].round(6)
        df = df[~df["Year_fraction"].isin(self.year_fraction_data(360)["Year_fraction"])]
        self.__data_rate = pd.merge(self.year_fraction_data(360),df,how='outer')
        self.__data_rate = self.__data_rate.sort_values(by='Year_fraction').reset_index(drop=True)
        return self.__data_rate

    def linear_interpol(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode linéaire.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__data_rate["Rate"].interpolate(method='linear')
        return self.__data_rate
    
    def quadratic_interpol(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode quadratic.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__data_rate["Rate"].interpolate(method='quadratic')
        return self.__data_rate

    def Nelson_Siegel_interpol(self,convention,product_year_fraction:list) -> pd.DataFrame:
        """
        Fonction pour interpoler les taux de la courbe de taux -> méthode Nelson-Siegel.
        """
        self.__data_rate = self.year_fraction_data(convention)
        Nelson_param = optimize_nelson_siegel(self.__data_rate["Year_fraction"],self.__data_rate["Rate"])
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        for rates in self.__data_rate["Year_fraction"]:
            if self.__data_rate.loc[self.__data_rate["Year_fraction"]==rates,"Rate"].isna().any():
                self.__data_rate.loc[self.__data_rate["Year_fraction"]==rates,"Rate"] = nelson_siegel(rates, Nelson_param[0], Nelson_param[1], Nelson_param[2], Nelson_param[3])
        return self.__data_rate

    def flat_rate(self,product_year_fraction: list) -> pd.DataFrame:
        """
        Fonction pour construire une courbe un taux fixe.
        """
        self.__data_rate = self.attribute_rates_curve(product_year_fraction)
        self.__data_rate["Rate"] = self.__flat_rate
        return self.__data_rate

    def forward_rate(self,product_year_fraction:list, type_interpol:str=TYPE_INTERPOL) -> pd.DataFrame:
        """
        Fonction pour calculer les taux forward de la courbe de taux.

        Input:
        - product_year_fraction: liste des maturités des produits
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        if type_interpol == "Linear":
            self.__data_rate = self.linear_interpol(product_year_fraction)
        if type_interpol == "Quadratic":
            self.__data_rate = self.quadratic_interpol(product_year_fraction)
        if type_interpol == "Nelson_Siegel":
            self.__data_rate = self.Nelson_Siegel_interpol(360,product_year_fraction)
        if type_interpol == "Flat":
            self.__data_rate = self.flat_rate(product_year_fraction)
        for i in range(len(self.__data_rate) - 1):
            year_fraction = self.__data_rate["Year_fraction"].iloc[i]
            next_year_fraction = self.__data_rate["Year_fraction"].iloc[i + 1]
            rate = self.__data_rate["Rate"].iloc[i]
            next_rate = self.__data_rate["Rate"].iloc[i + 1]
            self.__data_rate.at[i+1, "Forward_rate"] = ((((1 + next_rate) ** next_year_fraction) / ((1 + rate) ** year_fraction)) ** (1 / (next_year_fraction - year_fraction))) - 1
        return self.__data_rate
    
    def create_product_rate_curve(self,product_year_fraction: list, type_interpol:str=TYPE_INTERPOL) -> pd.DataFrame:
        """
        Fonction pour créer une courbe de taux pour un produit donné.

        Input:
        - product_year_fraction: liste des maturités des produits
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        self.__data_rate = self.forward_rate(product_year_fraction,type_interpol)
        self.__data_rate = self.__data_rate[self.__data_rate["Year_fraction"].isin(product_year_fraction)]

        return self.__data_rate
    
    def shift_curve(self, shift:dict, type_interpol:str=TYPE_INTERPOL):
        """
        Fonction pour shifter la courbe des taux, possibilité d'utiliser un shift linéaire ou non.
        Input:
        - shift: dictionnaire avec les clés = maturité des produits à shifter, valeurs = shift à appliquer pour chaque maturités
        - type_interpol: type d'interpolation à utiliser pour la courbe de taux (Linear, Quadratic, Nelson_Siegel, Flat)
        """
        product_year_fraction = shift.keys()
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        self.__data_rate['Rate']+=self.__data_rate['Year_fraction'].map(shift)
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        pass

    def deep_copy(self,flat_rate:float=None):
        return Rates_curve(self.__path_rate,flat_rate)
    
    def change_rate(self,product_year_fraction: list, fixed_rate, type_interpol = TYPE_INTERPOL) -> pd.DataFrame:
        self.__data_rate = self.create_product_rate_curve(product_year_fraction,type_interpol)
        self.__data_rate['Rate']= fixed_rate
        return self.__data_rate
    
    def return_data_frame(self):
        return self.__data_rate

    def Vasicek_volatility(self,curve_df,n_simulations) -> pd.DataFrame:
        """
        Fonction pour calculer la volatilité de la courbe de taux selon le modèle de Vasicek.
        """
        np.random.seed(123456)  # pour reproductibilité

        # Préparation des données
        r_t = curve_df["Rate"].values[:-1] / 100  # convertir en proportion
        r_t_dt = curve_df["Rate"].values[1:] / 100

        # Régression linéaire pour estimer alpha et beta
        X = r_t.reshape(-1, 1)
        y = r_t_dt - r_t
        model = LinearRegression().fit(X, y)
        alpha = model.coef_[0]
        beta = model.intercept_
        y_pred = model.predict(X)
        residuals = y - y_pred
        final_sigma = np.std(residuals)  # Volatilité empirique

        t = curve_df["Year_fraction"].values
        n_steps = len(t)

        # On recrée le vecteur de pas de temps (avec 0 pour le premier)
        delta_t_full = curve_df["Year_fraction"].diff().fillna(0.0).values

        # Estimation de k et theta
        k = -alpha/delta_t_full[1]
        theta = beta / k

        # Simulation de la courbe selon l'équation de Vasicek
        n_steps = len(curve_df)
        all_simulated_rates = np.zeros((n_simulations, n_steps))

        # Simulation de la courbe selon l'équation de Vasicek
        n_steps = len(curve_df)
        rates_simulated = np.zeros(n_steps)
        rates_simulated[0] = curve_df["Rate"].iloc[0] / 100  # point de départ

        for sim in range(n_simulations):
            rates_simulated = np.zeros(n_steps)
            rates_simulated[0] = curve_df["Rate"].iloc[0] / 100  # point de départ

            for i in range(1, n_steps):
                dt = delta_t_full[i]
                epsilon = np.random.normal(0, 1)
                r_prev = rates_simulated[i - 1]
                r_next = r_prev + k * (theta - r_prev) * dt + final_sigma * np.sqrt(dt) * epsilon
                rates_simulated[i] = r_next

            all_simulated_rates[sim] = rates_simulated

        '''# Tracé des simulations
        plt.figure(figsize=(10, 5))
        plt.plot(curve_df["Year_fraction"], curve_df["Rate"], label="Courbe observée", marker="o")
        for sim in range(n_simulations):
            plt.plot(curve_df["Year_fraction"], all_simulated_rates[sim] * 100, label=f"Simulation {sim + 1}", linestyle='--')
        plt.xlabel("Maturité (années)")
        plt.ylabel("Taux (%)")
        plt.title("Simulation de la courbe de taux - Modèle de Vasicek")
        plt.grid(True)
        plt.tight_layout()
        plt.show()'''

        # Retourner la volatilité et les courbes simulées
        return final_sigma, pd.DataFrame(all_simulated_rates * 100, columns=curve_df["Year_fraction"])

#Classe de recherche de la volatilité implicite
class ImpliedVolatilityFinder:
    """
    Class to find implied volatility on markets via BSM Model and different methods.
    Supported methods: Dichotomy, Optimization, Newton-Raphson.

    Inputs:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - option: Option object to be used for pricing.
    - price: Market Price of the option.
    - method: Method to be used for finding implied volatility (Dichotomy, Optimization, Newton-Raphson).
    - tolerance: Tolerance for the solver.
    - nb_iter: Max iterations for the solver.
    - bounds: Bounds for the solver (volatility).
    - starting_point: Starting point for the solver (volatility).
    - spot: Spot price of the underlying asset.
    """
    def __init__(self, model, option, price:float, method:str, tolerance:float, nb_iter:float, bounds:tuple, starting_point:float, spot:float) -> None:
        self._model=model
        self._option = option
        self._price = price
        self._method = method
        self._spot=spot

        self._tolerance = tolerance
        self._nb_iter = nb_iter
        self._bounds = bounds
        self._starting_point = starting_point

    def find_implied_volatility(self):
        try:
            method = IMPLIED_VOL_METHODS[self._method]
            return getattr(self, method)
        except KeyError:
            raise ValueError(f"Method {self._method} is not supported. Choose: {list(IMPLIED_VOL_METHODS.keys())}")
    
    @property
    def _dichotomy(self) -> float:
        """
        Find the implied volatility using the dichotomy method.
        """
        fct_vol=lambda volatility:self._model(self._option, volatility)
        low, high = self._bounds[0], self._bounds[1] 
        for _ in range(self._nb_iter):
            mid = (low+high)/2
            price = fct_vol(mid).price(self._spot)

            if abs(price - self._price) < self._tolerance:
                return mid
            elif price > self._price:
                high=mid
            else:
                low=mid
        return None

    @property
    def _optimization(self) -> float:
        """
        Find the implied volatility using optimization method.
        """
        fct_vol=lambda volatility:self._model(self._option, volatility)

        #Objective function:
        def objective(vol) -> float:
            price = fct_vol(vol[0]).price(self._spot)
            return (self._price - price)**2
        
        result = minimize(objective, self._starting_point, bounds=[self._bounds], method=SOLVER_METHOD, tol=self._tolerance, options={'maxfun': self._nb_iter})
        if result.success and result["fun"]<self._tolerance and result.x[0]!=self._starting_point:
            return result.x[0]
        else:
            if self._starting_point != 1.0:
                self._starting_point=1.0
                return self.__optimization
            else:
                print(f"Optimization failed. Method: {self._method}, Tolerance: {self._tolerance}, Max Iterations: {self._nb_iter}, Bounds: {self._bounds}, Starting Point: {self._starting_point}")
                return None
    
    @property
    def _newton_raphson(self) -> float:
        """
        Find the implied volatility using Newton-Raphson method.
        """
        loc_vol = self._starting_point
        fct_vol=lambda volatility:self._model(self._option, volatility)

        for _ in range(self._nb_iter):
            valo=fct_vol(loc_vol)
            price = valo.price(self._spot)
            vega = valo.vega(self._spot)

            if vega == 0:
                if self._starting_point<1.0:
                    self._starting_point+=0.1
                    return self._newton_raphson
                print("Zero Vega, cannot find implied volatility.")
                return None

            diff = price - self._price
            if abs(diff) < self._tolerance:
                return loc_vol
            loc_vol -= diff / vega

        print(f"Optimization failed. Method: {self._method}, Tolerance: {self._tolerance}, Max Iterations: {self._nb_iter}, Starting Point: {self._starting_point}")
        return None
        
#Classe pour trouver la volatilité paramétrique (vol fonction du strike)
class SVIParamsFinder:
    """
    Class to find SVI parameters on markets via SVI Model.

    Input:
    - model: Model used for options pricing (we advise BSM for fast executions).
    - vector_options: List of options to be used for pricing.
    - vector_prices: List of market prices of the options.
    - method_implied_vol: Method to be used for finding implied volatility (Dichotomy, Optimization, Newton-Raphson).
    - vector_spots: List of spot prices of the underlying assets.
    - tolerance: Tolerance for the solver.
    - nb_iter: Max iterations for the solver.
    - bounds: Bounds for the solver (volatility).
    - starting_point: Starting point for the solver (volatility).
    """
    def __init__(self, model, vector_options:list, vector_prices:list, method_implied_vol:str, spot:float, tolerance:float, nb_iter:float, bounds:tuple, starting_point:float, initial_svi:list=INITIAL_SVI, svi_method:str=SVI_SOLVER_METHOD) -> None:
        self._model=model
        self._options = vector_options
        self._prices = vector_prices
        self._market_prices = np.array(self._prices)
        self.T = self._options[0].T
        self._spot=spot

        self._method = method_implied_vol
        self._tolerance = tolerance
        self._nb_iter = nb_iter
        self._bounds = bounds
        self._starting_point = starting_point
        self._initial_svi = initial_svi
        self._svi_method=svi_method
        
        self.vector_implied_vol = self.__calculate_implied_vols()
    
    def __calculate_implied_vols(self) -> list:
        try:
            vols = []
            for i in range(len(self._options)):
                finder = ImpliedVolatilityFinder(self._model, self._options[i], self._prices[i], self._method, self._tolerance, self._nb_iter, self._bounds, self._starting_point, self._spot)
                vols.append(finder.find_implied_volatility())
            return vols
        except KeyError:
            raise ValueError("Error while calculating implied volatilities !")
    
    def find_svi_parameters(self):
        """
        Find SVI parameters for the given options matrice.
        """

        fct_vi=lambda a,b,p,m,s,k: a + b * (p * (k - m) + np.sqrt((k - m)**2 + s**2))
        fct_valo=lambda option, svi_vol: self._model(option, svi_vol)

        def objective(params)->float:
            a,b,p,m,s = params
            k_vec = np.array([np.log(option._strike/self._spot) for option in self._options])
            w_vec = fct_vi(a,b,p,m,s,k_vec)
            svi_vols = np.sqrt(w_vec/self.T)
            prices = [fct_valo(option, svi_vol).price(self._spot) for option, svi_vol in zip(self._options, svi_vols)]
            
            return np.sum(((prices - self._market_prices))**2)

        def param_constraint(params)-> float:
            a,b,p,m,s = params
            return a + b*s*np.sqrt(1+p**2)

        def check_params(params):
            a,b,p,m,s = params
            k_vec = np.array([np.log(option._strike/self._spot) for option in self._options])
            w_vec = fct_vi(a,b,p,m,s,k_vec)
            result = np.all(w_vec>0)
            return result

        def new_initial(bounds):
            perturbation = np.ones(len(self._initial_svi)) * np.random.uniform(-5,5)
            new_svi = self._initial_svi*perturbation
            for i in range(len(self._initial_svi)):
                if bounds[i][1] == None:
                    bounds[i][1] = 1000000
                if bounds[i][0] == None:    
                    bounds[i][0] = -1000000
            if all(new_svi[i] < bounds[i][1] and new_svi[i] > bounds[i][0] for i in range(len(self._initial_svi))):
                return new_svi
            else:
                return new_initial(bounds)

        counstraits=({'type': 'ineq','fun':param_constraint},)
        #bounds for (a,b,p,m,sigma)
        bounds = [[-5,5], [1e-12,5], [-0.999999, 0.999999], [-5,5], self._bounds]#for sigma self._bounds
        result = minimize(objective, x0=self._initial_svi, bounds=bounds, method=self._svi_method, constraints=counstraits, options=OPTIONS_SOLVER_SVI)
        count=1
        if result.success and check_params(result.x) == True and result["fun"]<50:
            return result.x
        elif count < 10:
            count +=1
            self._initial_svi = new_initial(bounds)
            return self.find_svi_parameters()
        else:
            print(f"Optimization failed. Method: {self._method}, Tolerance: {self._tolerance}, Max Iterations: {count}, Bounds: {self._bounds}, Starting Point: {self._starting_point}")
            return None

#Helper to get the market from the currency.
def get_market(currency):
    market_map = {
        "EUR": "XECB",
        "USD": "XNYS",
        "GBP": "IFEU",
        "BRL": "BVMF"
    }
    if currency is None:
        currency = BASE_CURRENCY

    market = market_map.get(currency)
    if market is None:
        raise ValueError(f"Currency '{currency}' is not supported! Choose from: {', '.join(market_map.keys())}")
    return market

#Helper to calculate the yield of a fixed-income product.
def calculate_yield(cashflows: dict, market_price:float, initial_rate:float=INITIAL_RATE):
    """
    Solve for Yield to Maturity (YTM) given a dictionary of cashflows.
    
    :param cashflows: Dictionary where keys are time in years and values are cashflows.
    :param initial_rate: Initial guess for YTM.
    :return: Yield to Maturity (YTM)
    """
    def ytm(y):
        return sum([cf["NPV"] / (1 + y) ** t for t, cf in cashflows.items()]) - market_price
    
    ytm_solution = fsolve(ytm, initial_rate)[0]
    return ytm_solution*100

#Helper to import heston pre-calibrated parameters
def get_heston_params_from_csv(pricing_date, path=os.path.join(DATA_PATH, "heston_params.csv")):
    from constants import get_from_cache, set_in_cache
    key = (FILE_PATH, FILE_UNDERLYING)
    cached = get_from_cache("HestonParams", key)
    if cached is not None:
        return cached

    df = pd.read_csv(path, index_col=0, sep=';')
    df.index = pd.to_datetime(df.index, format=FORMAT_DATE)
    match = df.loc[[pd.to_datetime(pricing_date, format=FORMAT_DATE)]]
    if match.empty:
        raise ValueError(f"No Heston parameters found for {pricing_date}")

    params = match.iloc[0][["v0", "kappa", "theta", "eta", "rho"]].to_dict()
    set_in_cache("HestonParams", key, params)
    return params