from controller import Robot

class MotorSim():

    def __init__(self, motor, sensor, direction, step_size):
        self.motor = motor
        self.sensor = sensor
        self.direction = direction
        self.sensor.enable(step_size)
        self.motor.enableTorqueFeedback(step_size)

    def set_position(self, pos):
        self.motor.setPosition(self.direction * pos)

    def get_position(self):
        return self.direction * self.sensor.getValue()

    def get_velocity(self):
        return self.direction * self.motor.getVelocity()

    def set_velocity(self, vel):
        self.motor.setVelocity(self.direction * vel)

    def get_torque(self):
        return self.direction * self.motor.getTorqueFeedback()

    def set_torque(self, tau):
        self.motor.setTorque(self.direction * tau)
