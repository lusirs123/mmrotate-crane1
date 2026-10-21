"""S048 own process watchdog; preserve historical GPU policies."""
import os
import threading
from crane_project.tools.audit_port_size_native_p3_v1 import smi,write

def setup_gpu(out,user_limit_mib=7000,process_stop_mib=6800,allocator_cap_mib=6000):
    if not 0<allocator_cap_mib<process_stop_mib<user_limit_mib:raise ValueError("Invalid GPU memory limits")
    cards=[r for r in smi('index,uuid,memory.free') if int(r[0]) in (0,1,2,3)]
    requested=os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None:
        if requested not in ('0','1','2','3'):raise ValueError('Only one available physical GPU authorized')
        cards=[r for r in cards if r[0]==requested]
    if not cards:raise RuntimeError('RESOURCE_STOP no authorized GPU')
    card=max(cards,key=lambda r:int(r[2]))
    if int(card[2])<3072:raise RuntimeError('RESOURCE_STOP free<3072MiB')
    os.environ['CUDA_VISIBLE_DEVICES']=card[0]
    state=dict(physical_gpu=int(card[0]),uuid=card[1],free_before_mib=int(card[2]),
               process_peak_polled_mib=0,process_stop_mib=process_stop_mib,allocator_cap_mib=allocator_cap_mib,user_limit_mib=user_limit_mib)
    stop=threading.Event()
    def watch():
        while not stop.is_set():
            try:
                apps=smi('pid,gpu_uuid,used_memory','compute-apps')
                own=sum(int(r[2]) for r in apps if r[0]==str(os.getpid()) and r[1]==card[1])
                state['process_peak_polled_mib']=max(own,state['process_peak_polled_mib'])
                if own>=process_stop_mib:
                    write(out/'RESOURCE_STOP.json',dict(state,reason='Own process reached%dMiB safety stop'%process_stop_mib))
                    os._exit(75)
            except Exception as e:
                # A blind monitor cannot promise the user cap: stop fail closed.
                write(out/'RESOURCE_STOP.json',dict(state,reason='GPU watchdog failed: '+str(e)))
                os._exit(75)
            stop.wait(.2)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    return state,stop,thread


