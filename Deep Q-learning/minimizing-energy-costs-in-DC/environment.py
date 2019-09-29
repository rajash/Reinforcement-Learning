import numpy as np
class Environment(object):
    # initializing the parameters of the environment
    def __init__(self,
                 optimal_temp = (18.0, 24.0),
                 initial_month = 0,
                 initial_number_users = 10,
                 number_users_range = (10,100),
                 max_update_users = 5,
                 initial_data_rate = 60,
                 data_rate_range = (20,300),
                 max_update_date = 5,
                 ):

        self.monthly_atmospheric_temps = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0,
                                                23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temp = self.monthly_atmospheric_temps[initial_month]
        self.optimal_temp = optimal_temp
        self.min_temp = -20
        self.max_temp = 80
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.min_number_users = number_users_range[0]
        self.max_number_users = number_users_range[1]
        self.max_update_users = max_update_users
        self.initial_data_rate = initial_data_rate
        self.current_rate_data = initial_data_rate
        self.min_data_rate = data_rate_range[0]
        self.max_data_rate = data_rate_range[1]
        self.max_update_data = max_update_date

        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users+ 1.25 * self.current_rate_data
        
        self.temp_ai = self.intrinsic_temp
        self.temp_noai = (self.optimal_temp[0] + self.optimal_temp[1])/ 2.0
        
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # updating the environment
    def update_env(self, direction, energy_ai, month):
        
        energy_noai = 0
        
        if (self.temp_noai < self.optimal_temp[0]):
            energy_noai = self.optimal_temp[0] - self.temp_noai
            self.energy_noai = self.optimal_temp[0]
        elif (self.temp_noai > self.optimal_temp[1]):
            energy_noai = self.temp_noai - self.optimal_temp[1]
            self.energy_noai = self.optimal_temp[1]

        self.reward = energy_noai - energy_ai

        self.reward = 1e-3 * self.reward

        self.atmospheric_temperature = self.monthly_atmospheric_temps[month] 

        self.current_number_users += np.random.randint(-self.max_number_users, self.max_number_users)

        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        if (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        
        self.current_rate_data += np.random.randint(-self.max_data_rate, self.max_data_rate)

        if (self.current_rate_data > self.max_data_rate):
            self.current_rate_data = self.max_data_rate
        if (self.current_rate_data < self.min_data_rate):
            self.current_number_users = self.min_data_rate

        past_intrinsic_temp = self.intrinsic_temp
        
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        
        delta_intrinsic_temp = self.intrinsic_temp - past_intrinsic_temp

        if(direction == -1):
           delta_temperature_ai = -energy_ai
        if(direction == 1):
           delta_temperature_ai = energy_ai
        
        self.temp_ai += delta_intrinsic_temp + delta_temperature_ai
        self.temp_noai += delta_intrinsic_temp

        if (self.temp_ai < self.min_temp):
            if(self.train == 1):
                self.game_over = 1
            else:
                self.temp_ai = self.optimal_temp[0]
                self.total_energy_ai += self.optimal_temp[0] - self.temp_ai
        elif (self.temp_ai > self.max_temp):
            if(self.train == 1):
                self.game_over = 1
            else:
                self.temp_ai = self.optimal_temp[1]
                self.total_energy_ai += self.temp_ai -self.optimal_temp[1]
        
        self.total_energy_ai += energy_ai
        self.total_energy_noai += energy_noai

        scaled_temp_ai = (self.temp_ai - self.min_temp)/(self.max_temp - self.min_temp)

        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)

        scaled_data_rate = (self.current_rate_data - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)

        next_state = np.matrix([scaled_temp_ai, scaled_number_users, scaled_data_rate])

        return next_state, self.reward, self.game_over

    # reseting the environment
    def reset(self, new_month):        
        self.atmospheric_temp = self.monthly_atmospheric_temps[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_data_rate
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temp_ai = self.intrinsic_temp
        self.temp_noai = (self.optimal_temp[0] + self.optimal_temp[1])/ 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # get the current state, last reward and whether the game is over or not
    def observe(self):
        scaled_temp_ai = (self.temp_ai - self.min_temp)/(self.max_temp - self.min_temp)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_data_rate = (self.current_rate_data - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)

        current_state = np.matrix([scaled_temp_ai, scaled_number_users, scaled_data_rate])

        return current_state, self.reward, self.game_over

    
