from controller import Robot

class MotorSim():

    def __init__(self, motor, sensor, direction, step_size):
        self.motor = motor
        self.sensor = sensor
        self.direction = direction
        self.sensor.enable(step_size)
        self.motor.enableTorqueFeedback(step_size)

    def setPosition(self, pos):
        self.motor.setPosition(self.direction * pos)

    def getPosition(self):
        return self.direction * self.sensor.getValue()

    def getVelocity(self):
        return self.direction * self.motor.getVelocity()

    def setVelocity(self, vel):
        self.motor.setVelocity(self.direction * vel)

    def getTorque(self):
        return self.direction * self.motor.getTorqueFeedback()

    def setTorque(self, tau):
        self.motor.setTorque(self.direction * tau)