

class GroupType:
    def __init__(self):
        self.group_name = self.__class__.__name__

    @property
    def group_name(self):
        return self.__group_name

    @group_name.setter
    def group_name(self, val):
        self.__group_name = val

    @property
    def group_id(self):
        return self.__group_id

    @group_id.setter
    def group_id(self, val):
        self.__group_id = val

    @property
    def is_manifold(self):
        return self.__is_manifold

    @is_manifold.setter
    def is_manifold(self, val):
        self.__is_manifold = val

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, val):
        self.__dim = val

    @property
    def manifold_dim(self):
        return self.__manifold_dim

    @manifold_dim.setter
    def manifold_dim(self, val):
        self.__manifold_dim = val

    @property
    def embedded_dim(self):
        return self.__embedded_dim

    @embedded_dim.setter
    def embedded_dim(self, val):
        self.__embedded_dim = val

class SO3Type(GroupType):
    def __init__(self):
        super().__init__()
        self.is_manifold = False
        self.group_id = 1
        self.dim = 4
        self.manifold_dim = 3
        self.embedded_dim = 4

class so3Type(SO3Type):
    def __init__(self):
        super().__init__()
        self.is_manifold = True
        self.dim = 3

class SE3Type(GroupType):
    def __init__(self):
        super().__init__()
        self.is_manifold = False
        self.group_id = 3
        self.dim = 7
        self.manifold_dim = 6
        self.embedded_dim = 7

class se3Type(SE3Type):
    def __init__(self):
        super().__init__()
        self.is_manifold = True
        self.dim = 6

SO3_type = SO3Type()
so3_type = so3Type()
se3_type = se3Type()
SE3_type = SE3Type()

SO3_type.mapping_type = so3_type
so3_type.mapping_type = SO3_type
SE3_type.mapping_type = se3_type
se3_type.mapping_type = SE3_type