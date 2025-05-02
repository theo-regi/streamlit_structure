import numpy as np
from scipy.stats import norm
from constants import OptionType, BarrierType, NUMBER_PATHS_H, NB_STEPS_H, BASE_DELTA_S, HESTON_PATHS_METHOD, SEED_SIMULATIONS
from datetime import datetime, timedelta
#-------------------------------------------------------------------------------------------------------
#----------------------------Script pour implémenter les différents models diffusion/pricing------------
#-------------------------------------------------------------------------------------------------------
#Black-Scholes-Merton model:
class BSM:
    """
    Black-Scholes-Merton model for pricing options.
    """
    def __init__(self, option, sigma:float=None) -> None:
        self._sigma = sigma
        self._option = option

    def d1(self, spot:float) -> float:
        return (np.log(spot/self._option._strike) + (self._option._rate - self._option._div_rate + (self._sigma**2)/2) * self._option.T) / (self._sigma * np.sqrt(self._option.T))

    def d2(self, spot:float) -> float:
        return self.d1(spot) - self._sigma * np.sqrt(self._option.T)

    def price(self, spot:float) -> float:
        """
        Calculate the price of the given option.
        """
        if self._option._type == OptionType.CALL:
            return spot * norm.cdf(self.d1(spot)) * np.exp(-self._option._div_rate*self._option.T) - self._option._strike * np.exp(-self._option._rate * self._option.T) * norm.cdf(self.d2(spot))
        elif self._option._type == OptionType.PUT:
            return self._option._strike * np.exp(-self._option._rate * self._option.T) * norm.cdf(-self.d2(spot)) - spot * norm.cdf(-self.d1(spot)) * np.exp(-self._option._div_rate*self._option.T)
        else:
            ValueError("Option type not supported !")
            pass

    def delta(self, spot:float) -> float:
        """
        Calculate Delta of the given option.
        """
        if self._option._type == OptionType.CALL:
            return norm.cdf(self.d1(spot)) * np.exp(-self._option._div_rate * self._option.T)
        elif self._option._type == OptionType.PUT:
            return (norm.cdf(self.d1(spot))-1) * np.exp(-self._option._div_rate * self._option.T)
        else:
            ValueError("Option type not supported !")
            pass

    def gamma(self, spot:float) -> float:
        """
        Calculate Gamma of the given option.
        """
        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-(self.d1(spot)**2)/2)
        return d1_prime * np.exp(-self._option._div_rate * self._option.T) / (spot * self._sigma * np.sqrt(self._option.T))

    def vega(self, spot:float) -> float:
        """
        Calculate Vega of the given option.
        """
        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-self.d1(spot)**2/2)
        return spot * np.sqrt(self._option.T) * d1_prime * np.exp(-self._option._div_rate * self._option.T)/100

    def theta(self, spot:float, rate:float=None) -> float:
        """
        Calculate Theta of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if rate is None:
            rate = self._option._rate
        q = rate - self._option._div_rate

        d1_prime = 1/np.sqrt(2*np.pi) * np.exp(-self.d1(spot)**2/2)

        if self._option._type == OptionType.CALL:
            return -(spot * np.exp(-self._option._div_rate * self._option.T) * d1_prime * self._sigma) / (2 * np.sqrt(self._option.T)) + q*spot*norm.cdf(self.d1(spot))* np.exp(-self._option._div_rate * self._option.T) - rate*self._option._strike*np.exp(-rate*self._option.T)*norm.cdf(self.d2(spot))
        elif self._option._type == OptionType.PUT:
            return -(spot * np.exp(-self._option._div_rate * self._option.T) * d1_prime * self._sigma) / (2 * np.sqrt(self._option.T)) - q*spot*norm.cdf(self.d1(spot))* np.exp(-self._option._div_rate * self._option.T) + rate*self._option._strike*np.exp(-rate*self._option.T)*norm.cdf(-self.d2(spot))
        else:
            ValueError("Option type not supported !")
            pass
        
    def rho(self, spot:float, rate:float=None) -> float:
        """
        Calculate Rho of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if rate is None:
            rate = self._option._rate
        if self._option._type == OptionType.CALL:
            return self._option._strike*self._option.T*np.exp(-rate*self._option.T)*norm.cdf(self.d2(spot))/100
        elif self._option._type == OptionType.PUT:
            return -self._option._strike*self._option.T*np.exp(-rate*self._option.T)*norm.cdf(-self.d2(spot))/100
        else:
            ValueError("Option type not supported !")
            pass

#Heston model:
class Heston:
    """
    Heston Model: for options pricing, preferably autocalls, vanilla call/puts (avoid for short term maturities).
    /!\ Recommanded for derivatives when using stocastic volatilities,
    /!\ Not recommanded for digits / barriers options.

    Input:
    - 
    """
    def __init__(self, option, params:dict, nb_paths:float=NUMBER_PATHS_H, nb_steps:float=NB_STEPS_H, paths_method:float=HESTON_PATHS_METHOD) -> None:
        self._paths_method=paths_method
        self._nb_paths=nb_paths
        self._nb_steps=nb_steps
        self._option=option
        self._params=params
        self._v0,self._kappa,self._theta,self._eta,self._rho,=list(params.values())

        self._payoffs=[]
        self._spots=[]
        self._price=None
        pass

    def price(self,spot:float)->float:
        """
        Calculate the price of the given option.
        """
        self.calculate_pay_offs(spot)
        price=np.exp(-self._option._rate*self._option.T) * np.mean(self._payoffs)
        if self._price is None:
            self._price=price
        return price

    def delta(self, spot:float, delta_p:float=BASE_DELTA_S) -> float:
        """
        Calculate Delta of the given option.
        """
        if self._price is not None:
            price=self._price
        else:
            price=self.price(spot)

        spot_prime=spot*(1+delta_p)
        price_prime=self.price(spot_prime)
        return (price_prime-price)/(spot_prime-spot)

    def gamma(self, spot:float, delta_p:float=BASE_DELTA_S) -> float:
        """
        Calculate Gamma of the given option.
        """
        if self._price is not None:
            price=self._price
        else:
            price=self.price(spot)

        spot_up=spot*(1+delta_p)
        spot_down=spot*(1-delta_p)

        price_up=self.price(spot_up)
        price_down=self.price(spot_down)
        return (price_up+price_down-2*price)/((spot_up-spot)**2)

    def vega(self, spot:float) -> float:
        """
        Calculate Vega of the given option.
        """
        if self._price is not None:
            price=self._price
        else:
            price=self.price(spot)

        new_model = self.__deep_copy__()
        new_model._params['v0']+=0.01
        new_model.__init__(new_model._option, new_model._params, new_model._nb_paths, new_model._nb_steps)
        price_prime = new_model.price(spot)
        return ((price_prime-price)/np.sqrt(0.01))/100

    def theta(self, spot:float, rate:float=None) -> float:
        """
        Calculate Theta of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if self._price is not None:
            price=self._price
        else:
            price=self.price(spot)

        new_option = self._option.__deep_copy__()
        date_obj = datetime.strptime(new_option._end_date, new_option._format)
        date_obj -= timedelta(days=1)
        new_date = date_obj.strftime(new_option._format)
        new_option._end_date=new_date

        if new_option._type == OptionType.CALL or new_option._type == OptionType.PUT:
            new_option.__init__(new_option._start_date, new_option._end_date, new_option._type, new_option._strike, new_option._rate, new_option._day_count, new_option._rolling_conv, new_option._notional, new_option._format, new_option._currency, new_option._price)
        elif new_option._type == BarrierType.CALL_DOWN_IN or new_option._type == BarrierType.CALL_DOWN_OUT or new_option._type == BarrierType.CALL_UP_IN or new_option._type == BarrierType.CALL_UP_OUT or new_option._type == BarrierType.PUT_DOWN_IN or new_option._type == BarrierType.PUT_DOWN_OUT or new_option._type == BarrierType.PUT_UP_IN or new_option._type == BarrierType.PUT_UP_OUT:
            new_option.__init__(new_option._start_date, new_option._end_date, new_option._type, new_option._strike, new_option._barrier_strike, new_option._rate, new_option._day_count, new_option._rolling_conv, new_option._notional, new_option._format, new_option._currency, new_option._div_rate, new_option._price)

        new_model = self.__deep_copy__()
        new_model._option=new_option
        new_model.__init__(new_model._option, new_model._params, new_model._nb_paths, new_model._nb_steps)

        dt = self._option.T - new_model._option.T
        new_price = new_model.price(spot)
        return (new_price-price)/dt
        
    def rho(self, spot:float, rate:float=None) -> float:
        """
        Calculate Rho of the given option, in case of using the option rate for the dividend, give a risk free rate.
        """
        if self._price is not None:
            price=self._price
        else:
            price=self.price(spot)

        if rate is None:
            rate = self._option._rate+0.01

        new_model = self.__deep_copy__()
        new_model._option._rate=rate

        price_prime = new_model.price(spot)
        return ((price_prime-price)/(rate-self._option._rate))/100

    def calculate_pay_offs(self, spot:float) -> None:
        """
        Computes the pay_off of the given option.

        Input: Spot price of the underlying asset.
        """
        if self._paths_method == 'Euler':
            paths=self._generate_paths_Euler(spot)
        elif self._paths_method == 'AES':
            paths=self._generate_paths_AES(spot)
        self._spots=paths['Spots'][:,-1]

        payoffs=[]
        for spot in self._spots:
            payoffs.append(self._option.payoff(spot))
        self._payoffs=payoffs
        pass

    def _generate_paths_AES(self, spot:float, seed:float=SEED_SIMULATIONS) -> np.ndarray:
        """
        Generate paths for the Heston model using the AES method.
        """
        if seed is not None:
            np.random.seed(seed)

        def CIR_Sample(nb_paths,kappa,eta,theta,s,t,v_s):
            delta = 4.0 *kappa*theta/eta/eta
            c= 1.0/(4.0*kappa)*eta*eta*(1.0-np.exp(-kappa*(t-s)))
            kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(eta*eta*(1.0-np.exp(-kappa*(t-s))))
            sample = c* np.random.noncentral_chisquare(delta,kappaBar,nb_paths)
            return  sample

        z1 = np.random.normal(0,1,size=(self._nb_paths, self._nb_steps))
        w1 = np.zeros((self._nb_paths, self._nb_steps+1))
        v = np.zeros((self._nb_paths, self._nb_steps+1))
        s = np.zeros((self._nb_paths, self._nb_steps+1))
        v[:,0] = self._v0
        s[:,0] = np.log(spot)

        t=np.zeros((self._nb_steps+1))
        dt = self._option.T/self._nb_steps

        for i in range(0, self._nb_steps):
            if self._nb_paths > 1:
                z1[:,i] = (z1[:,i] - np.mean(z1[:,i]))/np.std(z1[:,i])
            w1[:,i+1]=w1[:,i]+np.power(dt,0.5)*z1[:,i]

            v[:,i+1]=CIR_Sample(self._nb_paths, self._kappa, self._eta, self._theta, 0, dt, v[:,i])
            k0 = (self._option._rate - self._rho/self._eta*self._kappa*self._theta)*dt
            k1 = (self._rho*self._kappa/self._eta-0.5)*dt - self._rho/self._eta
            k2 = self._rho/self._eta
            s[:,i+1]=s[:,i]+k0 + k1*v[:,i] + k2*v[:,i+1] + np.sqrt((1-self._rho**2)*v[:,i])*(w1[:,i+1]-w1[:,i])
            t[i+1]=t[i]+dt

        sts = np.exp(s)
        paths = {"time":t, "Spots": sts}
        return paths

    def _generate_paths_Euler(self, spot:float, seed:float=SEED_SIMULATIONS) -> np.ndarray:
        """
        Generate paths for the Heston model with Euler method (more convergence in price with reduced number of paths.)
        """
        if seed is not None:
            np.random.seed(seed)

        z1 = np.random.normal(0,1,size=(self._nb_paths, self._nb_steps))
        z2 = np.random.normal(0,1,size=(self._nb_paths, self._nb_steps))
        w1 = np.zeros((self._nb_paths, self._nb_steps+1))
        w2 = np.zeros((self._nb_paths, self._nb_steps+1))
        v = np.zeros((self._nb_paths, self._nb_steps+1))
        s = np.zeros((self._nb_paths, self._nb_steps+1))

        v[:,0] = self._v0
        s[:,0] = np.log(spot)

        t=np.zeros((self._nb_steps+1))
        dt = self._option.T/self._nb_steps

        for i in range(0, self._nb_steps):
            if self._nb_paths > 1:
                z1[:,i] = (z1[:,i] - np.mean(z1[:,i]))/np.std(z1[:,i])
                z2[:,i] = (z2[:,i] - np.mean(z2[:,i]))/np.std(z2[:,i])
            z2[:,i] = self._rho * z1[:,i] + np.sqrt(1.0-self._rho**2)*z2[:,i]
            w1[:,i+1]=w1[:,i]+np.power(dt,0.5)*z1[:,i]
            w2[:,i+1]=w2[:,i]+np.power(dt,0.5)*z2[:,i]

            v[:,i+1]=v[:,i]+self._kappa*(self._theta-v[:,i])*dt + self._eta*np.sqrt(v[:,i])*(w1[:,i+1]-w1[:,i])
            v[:,i+1]=np.maximum(v[:,i+1],0.0)

            s[:,i+1]= s[:,i] + (self._option._rate - 0.5*v[:,i])*dt + np.sqrt(v[:,i])*(w2[:,i+1]-w2[:,i])
            t[i+1]=t[i]+dt
        
        sts = np.exp(s)
        paths = {"time":t, "Spots": sts}
        return paths

    def _generate_paths(self, spot:float, seed:float=SEED_SIMULATIONS) -> np.ndarray:
        """
        Generate paths for the Heston model using the specified method.
        """
        if self._paths_method == 'Euler':
            return self._generate_paths_Euler(spot, seed)
        elif self._paths_method == 'AES':
            return self._generate_paths_AES(spot, seed)
        else:
            raise ValueError("Unsupported path generation method: {}".format(self._paths_method))

    def __deep_copy__(self):
        """
        Create a deep copy of the Heston model.
        """
        new_option = self._option.__deep_copy__()
        return Heston(new_option, self._params, self._nb_paths, self._nb_steps)

#Local Volatility (Dupire Formula)    
class Dupire:
    """
    Dupire Local Volatility Model for Monte Carlo pricing.
    Requires a precomputed local volatility surface: sigma_loc(K, T)
    """
    def __init__(self, option, local_vol_surface_dupire, nb_paths:float=NUMBER_PATHS_H, nb_steps:float=NB_STEPS_H) -> None:
        """
        Parameters:
        - option: option object with standard fields (_strike, _rate, T, etc.)
        - local_vol_surface: callable or 2D interpolation of sigma_loc(K, T)
        - nb_paths: number of Monte Carlo paths
        - nb_steps: number of time steps
        """
        self._option=option
        self._local_vol=local_vol_surface_dupire
        self._first_date=list(self._local_vol._maturities_t.values())[0]
        self._first_strike=list(self._local_vol._strikes_supported)[1]
        self._nb_paths=nb_paths
        self._nb_steps=nb_steps

        self._payoffs=[]
        self._spots=[]
        self._price=None
        self._spread_vol=None
        pass

    def _generate_paths(self, spot:float, seed:float=SEED_SIMULATIONS) -> np.ndarray:
        """
        Generate paths for the Dupire model using the local volatility surface.
        """
        if seed is not None:
            np.random.seed(seed)
        
        z = np.random.normal(0, 1, size=(self._nb_paths, self._nb_steps))
        s = np.zeros((self._nb_paths, self._nb_steps+1))
        s[:,0]=np.log(spot)

        t=np.zeros((self._nb_steps+1))
        dt = self._option.T/self._nb_steps

        for i in range(0,self._nb_steps):
            check = False
            if self._nb_paths>1:
                z[:,i] = (z[:,i] - np.mean(z[:,i]))/np.std(z[:,i])
            effective_t = max(float(t[i]), (float(self._first_date)+0.0000001))
            effective_k = max(float(self._option._strike), float(self._first_strike))
            vol_loc = self._local_vol.get_local_implied_vol(effective_t, effective_k) + self._spread_vol
            count = 0
            while check == False and count < 1000:
                count += 1
                if vol_loc is None:
                    vol_loc = self._local_vol.get_local_implied_vol(effective_t, effective_k) + self._spread_vol
                else:
                    check = True
            s[:,i+1]=s[:,i] + (self._option._rate - 0.5*vol_loc**2) * dt + vol_loc * z[:,i] * np.sqrt(dt)          
            t[i+1]=t[i]+dt

        sts=np.exp(s)
        paths = {"time":t, "Spots": sts}
        return paths

    def _calculate_payoffs(self, spot:float, spread_vol:float=0) -> None:
        """
        Compute pay_offs of the given option.
        """
        self._spread_vol=spread_vol
        paths = self._generate_paths(spot)
        self._spots = paths['Spots'][:, -1]

        payoffs=[]
        for spot in self._spots:
            payoffs.append(self._option.payoff(spot))
        self._payoffs=payoffs
        pass

    def price(self, spot:float, spread_vol:float=0)->float:
        """
        Calculate the price of the given option using the local volatility surface.
        """
        self._calculate_payoffs(spot, spread_vol)
        price=np.exp(-self._option._rate*self._option.T) * np.mean(self._payoffs)
        if self._price is None:
            self._price=price
        return price

    def delta(self, spot:float, delta_p:float=BASE_DELTA_S):
        """
        Calculate Delta of the given option.
        """
        if self._price is None:
            self._price = self.price(spot)

        spot_prime=spot * (1 + delta_p)
        price_prime = self.price(spot_prime)
        return (price_prime - self._price)/(spot_prime-spot)

    def gamma(self, spot:float, delta_p=BASE_DELTA_S) -> float:
        """
        Calculate Gamma of the given option.
        """
        if self._price is None:
            self._price = self.price(spot)

        spot_up=spot*(1 + delta_p)
        spot_down=spot*(1 - delta_p)

        price_up=self.price(spot_up)
        price_down=self.price(spot_down)
        return (price_up+price_down-2*self._price) / ((spot_up-spot)**2)

    def vega(self, spot:float)-> float:
        """
        Calculate Vega of the given option.
        """
        if self._price is None:
            self._price = self.price(spot)

        #new_model = self.__deep_copy__()
        #new_model.__init__(new_model._option, new_model._local_vol, new_model._nb_paths, new_model._nb_steps, spread_vol=0.01)
        
        price_prime = self.price(spot,spread_vol=0.01)
        return ((price_prime - self._price)/np.sqrt(0.01))/100
    
    def theta(self, spot:float)->float:
        """
        Calculate Theta of the given option.
        """
        if self._price is None:
            self._price = self.price(spot)

        new_option=self._option.__deep_copy__()
        date_obj= datetime.strptime(new_option._end_date, new_option._format)
        date_obj-=timedelta(days=1)
        new_date=date_obj.strftime(new_option._format)
        new_option._end_date=new_date

        if new_option._type == OptionType.CALL or new_option._type == OptionType.PUT:
            new_option.__init__(new_option._start_date, new_option._end_date, new_option._type, new_option._strike, new_option._rate, new_option._day_count, new_option._rolling_conv, new_option._notional, new_option._format, new_option._currency, new_option._price)
        elif new_option._type == BarrierType.CALL_DOWN_IN or new_option._type == BarrierType.CALL_DOWN_OUT or new_option._type == BarrierType.CALL_UP_IN or new_option._type == BarrierType.CALL_UP_OUT or new_option._type == BarrierType.PUT_DOWN_IN or new_option._type == BarrierType.PUT_DOWN_OUT or new_option._type == BarrierType.PUT_UP_IN or new_option._type == BarrierType.PUT_UP_OUT:
            new_option.__init__(new_option._start_date, new_option._end_date, new_option._type, new_option._strike, new_option._barrier_strike, new_option._rate, new_option._day_count, new_option._rolling_conv, new_option._notional, new_option._format, new_option._currency, new_option._div_rate, new_option._price)

        new_model=self.__deep_copy__()
        new_model._option=new_option
        new_model.__init__(new_model._option, new_model._local_vol, new_model._nb_paths, new_model._nb_steps)

        dt=self._option.T-new_model._option.T
        new_price=new_model.price(spot)
        return (new_price-self._price)/dt
    
    def rho(self, spot:float, rate=None):
        price = self.price(spot)
        if rate is None:
            rate = self._option._rate + 0.01

        new_model = self.__deep_copy__()
        new_model._option._rate = rate

        price_prime = new_model.price(spot)
        return ((price_prime - price) / (rate - self._option._rate))/100

    def __deep_copy__(self):
        """
        Create a deep copy of the Heston model.
        """
        new_option = self._option.__deep_copy__()
        return Dupire(new_option, self._local_vol, self._nb_paths, self._nb_steps)
    