from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import time
from tqdm import tqdm
def worker(id, number, shm, arr_shm, sem):
    increased_number = 0

    for _ in tqdm(range(number)):
        increased_number += 1

    # 세마포어 획득
    sem.acquire()
    # 앞서 생성한 공유 메모리 블록을 가져와서 사용
    new_shm = shared_memory.SharedMemory(name=shm)
    # 가져온 공유 메모리 블록을 numpy 배열로 사용하기 편하게 변환
    tmp_arr = np.ndarray(arr_shm.shape, dtype=arr_shm.dtype, buffer=new_shm.buf)
    # 각각의 프로세스에서 연산한 값을 합해서 numpy 배열에 저장
    tmp_arr[0] += increased_number
    # 세마포어 해제
    sem.release()
    print(f'{id}번째 프로세스가 끝났습니다.')




if __name__ == "__main__":

    start_time = time.time()

    # 숫자를 저장할 numpy 배열(1차원) 생성
    arr = np.array([0])
    # 공유 메모리 생성
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    # 공유 메모리의 버퍼를 numpy 배열로 변환
    np_shm = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

    # 세마포어 생성
    sem = Semaphore()

    # 프로세스 2개 생성
    processes = []
    for i in range(24):
        p = Process(target=worker, args=(i, 800000000, shm.name, np_shm, sem))
        processes.append(p)
        p.start()
    # processes.append(Process(target=worker, args=(1, 500000000, shm.name, np_shm, sem)))
    # processes.append(Process(target=worker, args=(2, 500000000, shm.name, np_shm, sem)))
    # processes.append(Process(target=worker, args=(3, 500000000, shm.name, np_shm, sem)))
    # processes.append(Process(target=worker, args=(4, 500000000, shm.name, np_shm, sem)))
    # th2 = Process(target=worker, args=(2, 4, shm.name, np_shm, sem))
    # th3 = Process(target=worker, args=(3, 4, shm.name, np_shm, sem))
    # th4 = Process(target=worker, args=(4, 4, shm.name, np_shm, sem))

    # 프로세스 시작
    # for i in processes:
    #     i.start()


    # 프로세스가 종료될 때까지 기다린다.
    for p in processes:
        p.join()
    # th1.join()
    # th2.join()
    # th3.join()
    # th4.join()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("total_number=",end=""), print(np_shm[0])
    print("end of main")

    # 공유 메모리 사용 종료
    shm.close()
    # 공유 메모리 블록 삭제
    shm.unlink()
