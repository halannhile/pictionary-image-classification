from Bagger import Bagger

def run():
    for i in range(8):
        print("doing bag {}".format(i+1))
        bag = Bagger(8,7000)
        bag.run()
        bag.save_best("bagboost",i+1)

run()