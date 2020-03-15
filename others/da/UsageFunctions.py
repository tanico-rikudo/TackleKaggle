NUMERICS_TYPE = [int, float]
class UsageFunctions:
    def __init__(self):
        pass

    @staticmethod
    def message(message,strong=0):
        if strong == 0:
            print('{0}'.format(message),'\n')
        if strong == 1:
            print("-"*3,'{0}'.format(message),"-"*3,'\n')
        if strong == 2:
            print("="*3,'{0}'.format(message),"="*3,'\n')
        if strong == 10:
            print("[ERROR]:",'{0}'.format(message),'\n')
        if strong == 20:
            print("[DONE]:",'{0}'.format(message),'\n')
            
    @staticmethod
    def is_numarical(obj):
        return True if type(obj) in NUMERICS_TYPE else False