'''class SchoolMember(object):
    member_nums = 0
    def __init__(self, name, age, sex):
        self.name = name

        self.age = age
        self.sex = sex

    # self.enroll()

    def enroll(self):
        SchoolMember.member_nums += 1

        print("SchoolMember [%s] is enrolled!" % self.name)

    def tell(self):
        print("Hello my name is [%s]" % self.name)


class Teacher(SchoolMember):
    def __init__(self, name, age, sex, course, salary):  # 重写父类的__init__方法,正常
        super(Teacher, self).__init__(name, age, sex)  # 继承（新式类）

        SchoolMember.__init__(self,name,age,sex) #继承（旧式类）
        self.course = course
        self.salary = salary

    def teaching(self):
        print("Teacher [%s] is teaching [%s]" % (self.name, self.course))
'''


