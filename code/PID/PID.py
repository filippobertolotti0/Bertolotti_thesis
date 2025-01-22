class PID():
    def __init__(self) -> None:
        self.last_error = 0
        self.total_error = 0
        self.cumulative_error = 0
        self.kp = 0.1
        self.ki = 0
        self.kd = 220
        
    def predict(self, outputs, set_point):
        error = (outputs['temRoo.T'] - 273.15) - set_point
        self.total_error += (-error)
        delta_error = (-error) - self.last_error
        heat_P_power = outputs['heaPum.P']/5000
        
        control_signal = self.kp * (-error) + self.ki * 300 * self.total_error + (self.kd/300) * delta_error
        heat_P_power += control_signal
        control_signal = max(0, min(1, heat_P_power))
        
        self.last_error = -error
        self.cumulative_error += abs(error)
        
        return control_signal