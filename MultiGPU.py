import asyncio
import sys
import os
import subprocess
import signal
import datetime
import json
import time
import datetime
import requests
import logging
from multiprocessing import Queue

import GPUtil


class Director:
    """
    director만 반복하며 
    1) DB를 보고 taskQueue를 생성하고,
    2) workerList를 보고, gpu_avail할 때, worker생성
    3) timeout된 worker 강제 소멸
    """
    accomplish_set = set()
    done_set = set()
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    workers = []
    max_worker_count = os.cpu_count()*2-1

    def __init__(self, max_worker_count=None):
        """
        @type max_worker_count: int
        """
        if max_worker_count:
            self.max_worker_count = max_worker_count

    def pre_work(self):
        """
        시작전에 불완전 종료된 데이터에 대한 처리. 
        queue에 들어갔던 데이터들 다시 대기상태로
        """
        pass

    async def work(self):
        """
        반복해서 할 routine 작성
        """
        while True:

            # gpu avail check
            print('Current working workers => ', len(self.workers))
            async_task2 = asyncio.create_task(self.check_worker())
            self.after_process()
            gpu_id = self.get_available_device()
            # print('gpu_id =>',gpu_id)
            if gpu_id == -1:
                await asyncio.sleep(1)
                continue
            if len(self.workers) < self.max_worker_count and not self.tasks_to_accomplish.empty():
                # get from queue
                print("director will assign job to worker")
                job = self.tasks_to_accomplish.get_nowait()
                print("job => ", job)
                logging.info("job => %s", job)
                async_task = asyncio.create_task(
                    self.assign_worker(job, gpu_id))
                time.sleep(7)
                await asyncio.sleep(5)
                print('assigned worker => ', async_task)
                logging.info('assigned worker => %s', async_task)
            await asyncio.sleep(1)
            # upload to gcs

    async def assign_worker(self, job, gpu_id):
        print('assign worker')
        # assign worker here
        print('assign image worker')
        logging.info('assign image worker')
        async_task = asyncio.create_task(
            self.assign_img_worker(job, gpu_id))

        await async_task

        print('worker assigned')

    async def assign_img_worker(self, job, gpu_id):
        print('assign imageworker(job, gpu) =>', job, gpu_id)
        worker = ImgUpscaler(job, gpu_id)
        print('worker => ', worker)
        logging.info('image worker => %s', worker)
        self.workers.append(worker)
        print("img worker assigned")
        logging.info("img worker assigned")

    async def check_worker(self):
        poplist = []
        i = 0
        for worker in self.workers:
            if not worker.is_working:
                async_task = asyncio.create_task(worker.do_job())  # 이걸 task
                print(f'worker {worker.idx} start work')
                logging.info(f'worker {worker.idx} start work')
            if worker.get_status():
                self.accomplish_set.remove(worker.origin_task)
                self.done_set.add(worker.completed_task)
                self.tasks_that_are_done.put(worker.completed_task)
                print(f'worker {worker.idx} out')
                logging.info(f'worker {worker.idx} out')
                del worker
                poplist.insert(0, i)
            i += 1
        for popindex in poplist:
            self.workers.pop(popindex)
        # print('check worker finished')

        await asyncio.sleep(1)

    def get_available_device(max_memory=0.49):
        GPUs = GPUtil.getGPUs()
        freeMemory = 0
        available = -1
        max_memory = 0.64
        for GPU in GPUs:
            print('GPU memory util =>', GPU.memoryUtil)
            print('GPU memory free =>', GPU.memoryFree)
            print('GPU.memoryUtil type =>', type(GPU.memoryUtil))
            print('max_memory type =>', type(max_memory))
            if GPU.memoryUtil > max_memory:
                continue
            if GPU.memoryFree >= freeMemory:
                freeMemory = GPU.memoryFree
                available = GPU.id

        return available


class Worker:
    """
    base class
    """
    worker_count = 0

    def __init__(self, task, gpu_avail):
        print('workerbase init')
        self.work_done = False
        self.is_working = False
        # set timelimit
        self.timelimit = config.TIMEOUT
        Worker.worker_count += 1
        self.idx = Worker.worker_count
        self.start_dt = datetime.datetime.now()
        print('worker init at', self.start_dt)
        self.task = (*task, self.start_dt)
        self.origin_task = task
        self.gpu_avail = gpu_avail

    def get_status(self):
        """
        director가 worker의 상태를 점검하는 method
        완료 혹은 timeout 판별 
        True: work is finished or timeout
        False: in Processing
        """
        if self.work_done:
            print(f'worker_{self.idx} => job finished')
            return True
        elif (datetime.datetime.now() - self.start_dt) < self.timelimit:
            return False
        else:
            print(f'worker_{self.idx} => timeout')
            print(f'proc pid => {self.proc.pid}')
            self.completed_task = (*self.task, False)
            # self.proc.terminate()
            # self.proc.kill()
            os.killpg(os.getpgid(self.proc.pid + 1), signal.SIGTERM)
            os.system(f"/bin/kill -9 {self.proc.pid + 1}")
            print('worker killed')
            self.work_done = True
            return True

    def set_timelimit(self, minutes):
        """
        @type minutes: int
        @param minutes: minutes
        """
        self.timelimit = datetime.timedelta(minutes=minutes)


class ImgUpscaler(Worker):

    def __init__(self, task, gpu_avail):
        print('imgupscaler init')
        super().__init__(task, gpu_avail)

    async def do_job(self):
        self.is_working = True
        print(f'{self.idx} job start ')
        self.get_file_info()
        result = asyncio.create_task(self.do_upscale())
        print('job done. work result=', result)
        await asyncio.sleep(1)
        # self.work_done = True

    async def do_upscale(self):
        """
        여기서 GPU사용
        """
        print("start to upscale")
        logging.info("start to upscale")
        self.is_working = True
        cmd = F"""FFREPORT=file={SRLOG_DIR}/{self.id}_{self.sr_output_file_name}.report:level=32 {gpuscript} -progress pipe:1 -y -i {self.origin_file_path} -vf "eq=contrast=1,sr=dnn_backend=tensorflow:model={self.aimodel_path}:gpu_index={self.gpu_avail},hue=h={self.hue}:s={self.saturation}:b={self.brightness},convolution={self.sharpen}" {self.sr_image_fullpath} tile_size=300"""
        print(cmd)
        logging.info(cmd)
        self.proc = await asyncio.create_subprocess_shell(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', bufsize=0, shell=True, universal_newlines=False)
        # self.proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE ,encoding='utf-8', bufsize=0,shell=True)
        try:
            print("step2 - started upscaling id=",
                  self.task[0], f'-- [{datetime.datetime.now()}]')
            logging.info(
                f"step2 - started upscaling id={self.task[0]} -- [{datetime.datetime.now()}]")
            # r.update_progress(completed_task[1],10,True)
            sio.emit('send_event', {'event': 'image_progress_update',
                                    'userId': self.task[4], 'idx': self.task[0], 'progress': 10})
            ret = await self.proc.wait()
            # ret = await asyncio.wait_for(self.proc, 70.0)

        except:
            print("upscale fail")
            logging.warning("upscale fail")
        finally:
            try:
                print(
                    "step3 - ended upscaling,id={0},rc={1}".format(self.task[0], self.proc.returncode))
                logging.info(
                    "step3 - ended upscaling,id={0},rc={1}".format(self.task[0], self.proc.returncode))
                if self.proc.returncode == 0:
                    sio.emit('send_event', {'event': 'image_progress_update',
                                            'userId': self.task[4], 'idx': self.task[0], 'progress': 90})
                    complete_dt = datetime.datetime.now()
                    self.completed_task = (*self.task, complete_dt)  # OK
                    print('completed_task=>', self.completed_task)
                    logging.info('completed_task=> %s', self.completed_task)
                    self.work_done = True
                    return True
                # returncode가 0이 아니면 error처리
                else:
                    sio.emit('send_event', {
                             'event': 'image_progress_fail', 'userId': self.task[4], 'idx': self.task[0]})
                    dbConn_img.DBConn.set_image_TimeoutStatus(self.task[0])
                    self.completed_task = (*self.task, False)
                    print(f'task({self.task[0]}) failed')
                    logging.info(f'task({self.task[0]}) failed')
                    self.work_done = True
                    return False
            except:
                pass


class VidUpscaler(Worker):
    def __init__(self, task, gpu_avail):
        print('video upscaler init')
        logging.info('video upscaler init')
        super().__init__(task, gpu_avail)
        self.id = task[0]
        # vename, video_fullpath -> origin_file_path
        self.origin_file_path = task[1]
        self.aimodel_name = task[2]
        self.vename_splitext = os.path.splitext(
            self.origin_file_path.split('/')[-1])
        self.sr_output_file_name = self.vename_splitext[0] + \
            self.vename_splitext[1]
        self.sr_video_fullpath = MEDIA_ROOT + '/' + self.sr_output_file_name
        self.aimodel_path = AIMODEL_PATH + self.aimodel_name + '_3c.pb'

    async def do_job(self):
        self.is_working = True
        print(f'{self.idx} job start ')
        logging.info(f'{self.idx} job start ')
        self.get_file_info()
        result = asyncio.create_task(self.do_upscale())
        print('job done. work result=', result)
        logging.info('job done. work result= %s', result)
        await asyncio.sleep(1)
        # self.work_done = True

    def get_file_info(self):
        # 파일 정보가져오기
        print(f'worker {self.idx} get file_info')
        sio.emit('send_event', {"event": "video_start_video_upscaling",
                                "userId": self.task[3], 'idx': self.task[0]})
        batcmd = F"""/usr/local/bin/ffprobe -v quiet -print_format json -show_format -show_streams -select_streams v:0 {self.origin_file_path}"""
        result = json.loads(subprocess.check_output(batcmd, shell=True))
        self.codec = result["streams"][0]["codec_tag_string"].lower()
        self.video_time = result["format"]["duration"]
        self.video_duration = str(float(self.video_time)*100*1000)
        # self.cursor = dbConn_img.DBConn.getCursor() # 이건 왜?
        dbConn_video.DBConn.set_video_RunningStatus(
            self.id, self.sr_output_file_name, self.start_dt, self.video_duration)
        print('codec=>', self.codec)

    async def do_upscale(self):
        print("start to upscale")
        logging.info("start to upscale")
        self.is_working = True
        cmd = F"""FFREPORT=file={SRLOG_DIR}/{self.id}_{self.sr_output_file_name}.report:level=32 {gpuscript} -progress pipe:1 -y -i {self.origin_file_path} -vf "sr=dnn_backend=tensorflow:model={self.aimodel_path}:gpu_index={self.gpu_avail}" {self.sr_video_fullpath} tile_size=400"""
        print(cmd)
        logging.info(cmd)
        self.proc = await asyncio.create_subprocess_shell(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', bufsize=0, shell=True, universal_newlines=False, preexec_fn=os.setsid)
        # self.proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE ,encoding='utf-8', bufsize=0,shell=True, preexec_fn=os.setsid)
        await asyncio.sleep(5)
        try:
            print("step2 - started upscaling id=",
                  self.task[0], f'-- [{datetime.datetime.now()}]')
            logging.info(
                f"step2 - started upscaling id= {self.task[0]} -- [{datetime.datetime.now()}]")
            hashed_file_name = self.origin_file_path.split('/')[-1]
            hashkey = hashed_file_name.split(".")[0]
            print('hashkey =>', hashkey)
            # r.update_progress(hashkey,10,True)
            sio.emit('send_event', {'event': 'video_progress_update',
                                    'userId': self.task[3], 'idx': self.task[0], 'progress': 10})
            # ret = self.proc.wait()
            ret = await self.proc.wait()
            # ret = await asyncio.wait_for(self.proc, 70.0)
        except:
            print("upscale fail")
            logging.info("upscale fail")
        finally:
            try:
                print(
                    "step3 - ended upscaling,id={0},rc={1}".format(self.task[0], self.proc.returncode))
                logging.info(
                    "step3 - ended upscaling,id={0},rc={1}".format(self.task[0], self.proc.returncode))
                if self.proc.returncode == 0:
                    sio.emit('send_event', {'event': 'video_progress_update',
                                            'userId': self.task[3], 'idx': self.task[0], 'progress': 90})
                    complete_dt = datetime.datetime.now()
                    self.completed_task = (*self.task, complete_dt)  # OK
                    print('completed_task=>', self.completed_task)
                    logging.info('completed_task=> %s', self.completed_task)
                    self.work_done = True
                    return True
                # returncode가 0이 아니면 error처리
                else:
                    print('upscale fail')
                    logging.warning('upscale fail')
                    sio.emit('send_event', {
                             'event': 'video_progress_fail', 'userId': self.task[3], 'idx': self.task[0]})
                    dbConn_video.DBConn.set_video_TimeoutStatus(self.task[0])
                    self.completed_task = (*self.task, False)
                    print(f'task({self.task[0]}) failed')
                    logging.info(f'task({self.task[0]}) failed')
                    self.work_done = True
                    return False
            except:
                pass


async def main():
    director = Director(max_worker_count=2)
    lcpu = os.cpu_count()*2-1
    print('CPU=>', lcpu)
    director.pre_work()
    task1 = asyncio.create_task(director.work())
    print('task start')
    await task1
    body = 'upscaler asyncio finish'
    title = 'Upscaler stopped'
    send_mail(body=body, title=title)
    print('asyncio finish')
    logging.warning('asyncio finish')

try:
    asyncio.run(main())
except KeyboardInterrupt:
    body = 'stopped by keyboard interrupt'
    title = 'Upscaler stopped'
    send_mail(body=body, title=title)
    print('finish')
    logging.warning('finish')
except Exception as e:
    body = f'stopped by {e}'
    title = 'Upscaler stopped'
    send_mail(body=body, title=title)
