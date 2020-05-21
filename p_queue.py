import heapq
import numpy as np
import itertools
 
class Pqueue:
    def __init__(self):
        self.q=[]

    def pop(self):
        return heapq.heappop(self.q)

    def push(self, p, el):
        for e in self.q:
            if(self._equal(e[1], el)):
                if(p < e[0]):
                    self.q.remove(e)
                    heapq.heapify(self.q)
                    heapq.heappush(self.q, (p,el))
                    return
        heapq.heappush(self.q, (p,el))

    def empty(self):
        return len(self.q) == 0

    def len(self):
        return len(self.q)

    def _equal(self, SA1, SA2):
        return np.any(np.equal(SA1[0],SA2[0])) and SA1[1] == SA2[1]


class PQueue:
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        add_to_q = True
        if task in self.entry_finder:
            add_to_q = self.remove_task_if_lower_priority(task, priority)
        if add_to_q:
            count = next(self.counter)
            entry = [priority, count, task]
            self.entry_finder[task] = entry
            heapq.heappush(self.pq, entry)

    def remove_task_if_lower_priority(self, task, priority):
        'Mark an existing task as self.REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        if entry[0] > priority:
            entry[-1] = self.REMOVED
            return True
        else:
            return False

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                print(task)
                print(self.entry_finder)
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return len(self.entry_finder) == 0

    def qsize(self):
        return len(self.entry_finder)

def test():
    q = PQueue()
    q.add_task((tuple(np.array([1,2,3])),1),1)
    q.add_task((tuple(np.array([1,2,3])),1),2)
    q.add_task((tuple(np.array([4,5,6])),1),-1)
    q.add_task((tuple(np.array([1,2,3])),1),-10)

    while(not q.empty()):
        print(q.qsize())
        print("POP:",q.pop_task())
