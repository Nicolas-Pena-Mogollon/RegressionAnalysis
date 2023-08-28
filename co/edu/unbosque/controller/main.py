from co.edu.unbosque.controller.Controller import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        exit()
