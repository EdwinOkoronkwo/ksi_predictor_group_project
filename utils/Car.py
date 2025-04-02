class Car:
    """
    Represents a car with VIN, make, and model, and provides registration functionality.
    """

    _registered_vins = []  # Class attribute to store registered VINs

    def __init__(self, vin, make, model):
        """
        Initializes a Car object.
        """
        if not isinstance(vin, str) or not isinstance(make, str) or not isinstance(model, str):
            raise TypeError("VIN, make, and model must be strings.")

        if vin in Car._registered_vins:
            raise ValueError(f"VIN '{vin}' is already registered.")

        self.__vin = vin
        self.__make = make
        self.__model = model
        self.registered = False
        Car._registered_vins.append(vin)

    @property
    def vin(self):
        """Returns the VIN."""
        return self.__vin

    @property
    def make(self):
        """Returns the make."""
        return self.__make

    @property
    def model(self):
        """Returns the model."""
        return self.__model

    def register(self):
        """Registers the car."""
        self.registered = True

    def __bool__(self):
        """
        Overrides the boolean operator to check if the car is registered.
        """
        return self.registered

    @classmethod
    def get_count_registered_cars(cls):
        """
        Returns the number of registered cars.
        """
        return len(cls._registered_vins)

def main():
    """Main program to demonstrate Car class usage."""
    try:
        car1 = Car("VIN123", "Audi", "A4")
        print(f"Car created: {car1.make} {car1.model} (VIN: {car1.vin})")

        car1.register()
        print(f"Car registered: {car1.make} {car1.model} (VIN: {car1.vin})")

        print(f"Car1 registered status: {bool(car1)}")

        car2 = Car("VIN456", "Audi", "A5")
        print(f"Car created: {car2.make} {car2.model} (VIN: {car2.vin})")

        print(f"Total registered cars: {Car.get_count_registered_cars()}")

        try:
            Car(123, "Audi", "Quattro")
        except TypeError as e:
            print(f"Error creating car: {e}")

        try:
            Car("VIN123", "Audi", "A6")
        except ValueError as e:
            print(f"Error creating car: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()