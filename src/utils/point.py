class Point:
    """
    Represents a point in 2 dim space.
    """
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def scale(self, scale_factor: float):
        """
        Scales both coordinates with the given scale_factor
        """
        self.x = round(self.x*scale_factor)
        self.y = round(self.y*scale_factor)


    def scalex(self, scale_factor: float):
        self.x = round(self.x*scale_factor)


    def scaley(self, scale_factor: float):
        self.y = round(self.y*scale_factor)

    def set_from_string(self, coords: str, sep: str = ','):
        """
        Sets the coordinates according to the String coords assuming the strucutre: 'xsepy'.
        Example: coords = '13,12' amd sep = ','
        :param coords: Coordinate String
        :param sep: Seperator
        """
        self.x = int(coords.split(sep)[0])
        self.y = int(coords.split(sep)[1])

    def get_as_list(self) -> list:
        """
        Returns the coordinates as a two dimensional list
        """
        return [self.x, self.y]

    def __str__(self):
        return str(round(self.x)) + ',' + str(round(self.y))