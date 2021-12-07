from real_time.LatentWalkerController import LatentWalkerController

if __name__ == "__main__":
    controller = LatentWalkerController()
    while True:
        value = input()
        if value == "q":
            exit()
        print(type(value))
        controller.latent_walk.class_idx = int(value)
