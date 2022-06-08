from os.path import splitext
import csv

class CarBase:
    def __init__(self, photo_file_name=None, brand=None, carrying=None):
        self.photo_file_name = photo_file_name
        self.brand = brand
        self.carrying = float(carrying)

    def get_photo_file_ext(self):
        information = splitext(self.photo_file_name)[-1]
        return information


class Car(CarBase):
    def __init__(self, brand=None, photo_file_name=None, carrying=None, passenger_seats_count=None):
        super(Car, self).__init__(photo_file_name, brand, carrying)
        self.passenger_seats_count = int(float(passenger_seats_count))
        self.car_type = Car.__name__.lower()


class Truck(CarBase):
    @classmethod
    def string_parser(cls, string: str):
        result_array = string.split('x')
        err_status = True
        if len(result_array) != 3:
            err_status = False

        if err_status:
            for result_buff in result_array:
                if result_buff != '':
                    try:
                        float(result_buff)
                    except:
                        err_status = False
                        break
                else:
                    err_status = False
                    break

        try:
            if not err_status:
                raise IndentationError
        except IndentationError:
            return ['0', '0', '0']
        return result_array

    def __init__(self, brand=None, photo_file_name=None, carrying=None, body_whl=None):
        super(Truck, self).__init__(photo_file_name, brand, carrying)
        sizes_array = Truck.string_parser(body_whl)
        self.body_width = float(sizes_array[1])
        self.body_height = float(sizes_array[2])
        self.body_length = float(sizes_array[0])
        self.car_type = Truck.__name__.lower()

    def get_body_volume(self):
        return self.body_length * self.body_width * self.body_height


class SpecMachine(CarBase):
    def __init__(self, brand=None, photo_file_name=None, carrying=None, extra=None):
        super(SpecMachine, self).__init__(photo_file_name, brand, carrying)
        self.extra = extra
        self.car_type = 'spec_machine'


def get_car_list(csv_filename):
    obj_arrays = []
    CAR_LIT = [0, 1, 2, 3, 5]
    TRUCK_LIT = [0, 1, 3, 4, 5]
    SPEC_LIT = [0, 1, 3, 5, 6]

    def valid_check(rows):
        if not rows:
            return False
        if '' in rows:
            return False
        else:
            return True

    def type_check(rows):
        try:
            if rows[0] == 'car':
                return [rows[i] for i in CAR_LIT]
            if rows[0] == 'truck':
                TRUCK_COSTIL = [rows[i] for i in TRUCK_LIT]
                if TRUCK_COSTIL[-2] == '':
                    TRUCK_COSTIL[-2] = '0x0x0'
                return TRUCK_COSTIL
            if rows[0] == 'spec_machine':
                return [rows[i] for i in SPEC_LIT]
        except:
            return None
    car_list = []
    with open(csv_filename) as store:
        reader = csv.reader(store, delimiter=';')
        next(reader)
        for row in reader:
            car_list.append(row)
    normal_list = list(map(type_check, car_list))
    valid_list = list(filter(valid_check, normal_list))
    for obj in valid_list:
        if obj[0] == 'car':
            if (obj[3].__contains__('.')) and (obj[3].split('.')[0] != '') and (len(obj[3].split('.')) == 2):
                obj_arrays.append(Car(obj[1], obj[3], obj[4], obj[2]))

        if obj[0] == 'truck':
            if (obj[2].__contains__('.')) and (obj[2].split('.')[0] != '') and (len(obj[2].split('.')) == 2):
                obj_arrays.append(Truck(obj[1], obj[2], carrying=obj[4], body_whl=obj[3]))

        if obj[0] == 'spec_machine':
            if (obj[2].__contains__('.')) and (obj[2].split('.')[0] != '') and (len(obj[2].split('.')) == 2):
                obj_arrays.append(SpecMachine(obj[1], obj[2], obj[3], obj[4]))
    return obj_arrays
'''
car = Car('Bugatti Veyron', 'bugatti.png', '0.312', '2.0')
print(car.car_type, car.brand, car.photo_file_name, car.carrying, car.passenger_seats_count, sep='\n')
print('------')
truck = Truck('Nissan', 'nissan.jpeg', '1.5', '3.92x2.09x1.87')
print(truck.car_type, truck.brand, truck.photo_file_name, truck.body_length, truck.body_width, truck.body_height, sep='\n')
print('-----')
spec_machine = SpecMachine('Komatsu-D355', 'd355.jpg', '93', 'pipelayer specs')
print(spec_machine.car_type, spec_machine.brand, spec_machine.carrying, spec_machine.photo_file_name, spec_machine.extra, sep='\n')
print('-----')
print(spec_machine.get_photo_file_ext())

cars = get_car_list('cars_test.csv')
print(len(cars))
for car in cars:
    print(type(car))

print(cars[0].passenger_seats_count)
print(cars[1].get_body_volume())
'''