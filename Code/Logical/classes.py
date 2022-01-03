from enum import Enum

class ImageClasses(Enum):
    Bart = 0
    Homer = 1
    Lisa = 2
    Marge = 3
    Unknown = 4

class FaceClasses(Enum):
    Face = 1
    NoFace = -1