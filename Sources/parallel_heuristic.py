from multiprocessing import Manager, Process, Event
import multiprocessing as mp
from numpy import random
import time
import shutil
from common import UB, incumbent, ProblemData
from greedy_search import greedy_search
from rins_heuristic import rins_heuristic
import random as std_random
import os

GS_COPIES=5
RINS_COPIES=1

dct = None

def init(data, start_solution, ub):
	global dct
	manager = mp.Manager()
	dct = manager.dict()
	term_event = mp.Event()
	new_incumbent_event = mp.Event() 

	dct['UB'] = ub
	dct['incumbent'] = start_solution.copy()
	dct['obj_vector'] = {e : 0 for e in data.G.edges}

	gs_pool = []
	for i in range(GS_COPIES):
		if i == 0:
			start_solution = {e : 0 for e in data.G.edges}
		else:
			start_solution = {e : std_random.randint(0,1) for e in data.G.edges}

		worker_gs = mp.Process(target=do_work_greedy_search, name='myWorkerGS', 
			args=(dct, new_incumbent_event, data, start_solution), daemon = True)
		worker_gs.start()
		gs_pool.append(worker_gs)

	rins_pool = []
	for i in range(RINS_COPIES):
		worker_rins = mp.Process(target=do_work_rins, name='myWorkerRINS', 
			args=(dct, new_incumbent_event, data), daemon = True)
		worker_rins.start()
		rins_pool.append(worker_rins)

	return gs_pool, rins_pool, dct, new_incumbent_event
	
def getUB():
	return dct['UB']

def getIncumbentSolution():
	return dct['incumbent']

def setUBnIncumbent(newUB, incumbent, verbose=False, sync_event=None):
	global dct
	UB = dct['UB']
	if UB > newUB:
		dct['UB'] = newUB
		dct['incumbent'] = incumbent.copy()
		non_zero_edges = [e for e, val in incumbent.items() if val > 0.5]
		if sync_event:
			sync_event.set()
		if verbose:
			source = mp.current_process().name
			print(f'==================\n[{source}] Parallel heuristic solver: new UB={newUB} from process={os.getpid()}\n==================')


def do_work_greedy_search(dct, sync_event, data, start_solution):
	print(f'==================\nHello from the Parallel heuristic solver:: Greedy Search,  UB={dct["UB"]}, PID={os.getpid()}\n==================')
	print(f'==================\n Parallel heuristic solver:: Greedy Search started\n==================')
	greedy_search(data, start_solution, setUBnIncumbent, True, dct, sync_event)

def do_work_rins(dct, sync_event, data):
	print(f'==================\nHello from the Parallel heuristic solver::RINS,  UB={dct["UB"]}, PID={os.getpid()}\n==================')
	print(f'==================\n Parallel heuristic solver:: RINS started\n==================')
	rins_heuristic(data, setUBnIncumbent, False, dct, sync_event)
