from queue import PriorityQueue

q = PriorityQueue()

q.put((-1, (3,5)))
q.put((1, 'eat'))
q.put((3, 'sleep'))

while not q.empty():
    next_item = q.get()
    print(next_item)
