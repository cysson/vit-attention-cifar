import sys
import time

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # reset for new bar

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    bar = '[' + '=' * cur_len + '>' + '.' * rest_len + ']'

    sys.stdout.write(bar)

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    msg_list = []
    msg_list.append(f' Step: {step_time:.2f}s')
    msg_list.append(f' | Tot: {tot_time:.2f}s')
    if msg:
        msg_list.append(f' | {msg}')

    msg_str = ''.join(msg_list)
    sys.stdout.write(msg_str)

    sys.stdout.write(' ' * 10)
    sys.stdout.write(f'\r')

    if current == total - 1:
        sys.stdout.write('\n')
    sys.stdout.flush()
