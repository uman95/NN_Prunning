import time

from Baseline.main import main



"""
 This Runs The Baseline for each model we are working with. 
"""
if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("====> total time: {}m {:.2f}s".format(elapsed_time//60, elapsed_time%60))