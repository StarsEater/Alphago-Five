import time 
def func2(args):
    x,y = args[0],args[1]
    time.sleep(1)
    return x-y 

if __name__ == '__main__':
    from multiprocessing import Pool 
    cpu_worker_num = 3
    process_args = [(1,1),(9,9),(4,4),(3,3)]
    print(f'|inputs: {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outs = p.map(func2, process_args)
    