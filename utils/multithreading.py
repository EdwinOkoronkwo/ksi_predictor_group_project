import threading, time
def sleeper(t, name):
  print(f'{name} going to sleep for {t}s.')
  time.sleep(t)
  print(f'{name} has awoken from sleep.')
t1 = threading.Thread(target = sleeper, args = (5, 'Thread 1'))
t2 = threading.Thread(target = sleeper, args = (3, 'Thread 2'))
t1.start()
t2.start()
print(f'{threading.active_count()} active threads.')
print('Program has terminated.')