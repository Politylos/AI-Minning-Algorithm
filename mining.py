#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).
,

Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import collections
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools # @lru_cache(maxsize=32)

from numbers import Number

from search import Node, memoize, Queue,LIFOQueue,FIFOQueue,PriorityQueue,Problem,graph_search,breadth_first_graph_search,depth_first_graph_search
import time
import search


Node.expand = functools.lru_cache(maxsize=32)(Node.expand) 
Node.child_node = functools.lru_cache(maxsize=32)(Node.child_node)

def my_team():
    '''
    Returns
    List of team members in the form (student_number, first_name, last_name)        
    '''
    return [("10489045","Sophia", "Politylo")]
    
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    




class Mine(Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
        self.shape: tuple of mine shape found with np.shape
        self.top: array of zeros to be added to top of underground array to represent an un-dug state
        self.Max_payoff_depth: max depth before all payoffs are negitive
        self.initial: un dug mine state of the mine (aka. intial mine state)
        self.excavated: number of cells dug in best state of the mine
        self.maxpayoff: payoff for the best state of the mine (aka. max payoff the mine can achive while being valid)
        self.is3d: True(3d Mine) or False(2d Mine) bool, used to save computation on if the mine is 3d or 2d
        self.best_actionpay: holistic array of all actions possible in the mine, precomputed taking into acount neighbouring cells below and action cell
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, self.initial, self.shape,self.top,self.Max_payoff_depth,self.excavated,
        self.maxpayoff and self.is3d
        

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''
        self.dig_tolerance = dig_tolerance;     #tolerance of diffrence between cells
        self.underground = convert_to_tuple(underground);   #mine payoff for each cell in tuple
        self.len_x = underground.shape[0]   #number of x collums 
        self.top = np.zeros(np.append(self.len_x,1))    #array of zeros to append to top of mine for no dig
        self.len_y = 0;     #number of rows 0 for 2d mine
        self.mine_shape = underground.shape     #shape of mine for easy indexing
        
        #check if the mine has only a single column 
        if len(self.mine_shape) == 1:
            self.len_z = underground.shape[0]   #depth of mine
            self.len_x = 1
            self.cumsum_mine = convert_to_tuple(np.cumsum(underground)) # cumilitive sum off each collum in the underground array
            self.Max_payoff_depth = tuple([self.mine_shape,self.len_z-1,])    #depth of the deepest max payoff in each cumsum column
        else:
            self.len_z = underground.shape[1]; #depth of mine
            self.cumsum_mine = convert_to_tuple(np.cumsum(underground,axis=1))   # cumilitive sum off each column in the underground array
            self.Max_payoff_depth = convert_to_tuple(np.full(self.mine_shape[:-1],self.len_z-1))  #depth of the deepest max payoff in each cumsum column
        self.initial = convert_to_tuple(np.full(self.len_x,-1)) #intital mine state shape of x or x,ywith -1 representing not dug yet
        self.excavated=0;   #store to check if a state is more effective than the current best state
        self.maxpayoff=-1;  #best mine state found payoff
        self.x_pos = tuple(np.linspace(0,self.len_x-1,self.len_x,dtype=np.intp))    # list of all indexs for the x dimension 
        if (underground.ndim==3):
            self.len_y = underground.shape[1]; # number of y for a 3d mine
            self.len_z = underground.shape[2] #depth in a 3d mine
            self.top = np.zeros((self.len_x,self.len_y,1)) #top cells for an un-dug mine to add to payoff
            self.y_pos = tuple(np.tile(np.linspace(0,self.len_y-1,self.len_y,dtype=np.intp),self.len_x)) #list of all indexs for a 3d mine
            self.initial = convert_to_tuple(np.full((self.len_x,self.len_y),-1)) #inital mine state for a 3d mine 
            self.cumsum_mine = convert_to_tuple(np.cumsum(underground,axis=2)) #cumulative sum of a 3d mines columns 
            self.x_pos = tuple(np.repeat(np.linspace(0,self.len_x-1,self.len_x,dtype=np.intp),self.len_y)) #list of all x indexs for a 3d mine
        self.is3d = self.not2D() #store if mine is 3d or 2d, true or false
        self.Output_gen() #pre-computed holistic for each cell in the underground to form the Max_payoff_depth for the bb algorithm
            
        
    def Output_gen(self):
        """
        logic to choose correct function to generate the holistic for cutoff
        """
        if self.is3d:
            self.Output_gen3d()
        elif self.len_x ==1:
            self.Output_gen2d1d()
        else:
            self.Output_gen2d()
    def Output_gen3d(self):
        """

        Returns
        -------
        None.
        
        Creates a holisic for a 3D mine to trim actions in BB
        creates an admistable holistic for a 3d mine, with each cell Representing an actions evental payoff
        first checks if action will result in a postive payoff in the newly dug cell, returns it as a valid action
        else checks if it neighbours directly below are postive and have a greater payoff than is lost if true it is a valid action
        else checks neigbours directly below : +1 until the max depth or a positive payoff is found
        after taking into account all losses and gains for getting to that cell, if a postive payoff is found it is a valid action
        
        valid actions have a postive value 
        values to trim have a negitive value
        
        hol: holisitc to append for an action into action array
            negitive of bad action postive if good action
        """
        
        #empty array for corresponding holisitc
        self.best_actionpay = np.zeros(self.mine_shape)
        underground = np.array(self.underground)
        #check each cell in underground
        for cor, pay in np.ndenumerate(underground):
            #cor  = coordinate of cell in underground
            #pay = payoff of digging that cell in underground
            #allows a postive and valid path if cell has a postive pay off
            if pay > 0 or cor[2] == self.len_z-1:
                hol = pay #return postive payoff for action digging this cell
            else:
                #constructing a valid Slice to grab all neighbours bellow and to the side and bellow the current negitive payoff cell
                zstart  =cor[2]+1
                if cor[0] == 0:
                    xstart = cor[0];
                    xend = cor[0]+2
                elif cor[0] == self.len_x-1:
                    xstart = cor[0];
                    xend = cor[0]+2
                else:
                    xstart = cor[0]-1;
                    xend = cor[0]+2

                if cor[1] == 0:
                    ystart = cor[1];
                    yend = cor[1]+2
                elif cor[1] == self.len_x-1:
                    ystart = cor[1];
                    yend = cor[1]+2
                else:
                    ystart = cor[1]-1;
                    yend = cor[1]+2
                zend = zstart+1
                hol=-1; #set init value to 0 to start while loop
                #starts by checking cells directly bellow and neighbouring the cell expending down untill the bottom of the mine is reached or a postive holistic is reached
                while(zend < self.len_z+1 and hol < 0):
                    Slice = np.s_[xstart:xend,ystart:yend,zstart:zend]
                    if zend-1 == zstart:
                        values = underground[Slice]
                        posvalue = values[values > 0]
                        if len(posvalue) > 0:
                            hol=pay+np.sum(posvalue) #check if cells below are good to dig
                        else:
                            hol=pay+np.sum(values) #checks if any combination of cells below give a postive payoff
                    else:
                        hol=pay+np.sum(underground[Slice]) #checks if any combination of cells below give a postive payoff
                    zend+=1
            self.best_actionpay[cor] = hol #append holistic into master array
        self.best_actionpay=np.append(self.top,self.best_actionpay,axis=-1) #adds 0 to top of array as the no dig holistic
    def  Output_gen2d1d(self):
        """

        Returns
        -------
        None.
        
        Creates a holisic for a 1D mine to trim actions in BB
        creates an admistable holistic for a 3d mine, with each cell Representing an actions evental payoff
        first checks if action will result in a postive payoff in the newly dug cell, returns it as a valid action
        else checks if it neighbours directly below are postive and have a greater payoff than is lost if true it is a valid action
        else checks neigbours directly below : +1 until the max depth or a positive payoff is found
        after taking into account all losses and gains for getting to that cell, if a postive payoff is found it is a valid action
        
        valid actions have a postive value 
        values to trim have a negitive value
        
        hol: holisitc to append for an action into action array
            negitive of bad action postive if good action

        """
        #empty array for corresponding holisitc
        self.best_actionpay = np.zeros(self.mine_shape)
        underground = np.array(self.underground)
        #check each cell in underground
        for cor, pay in np.ndenumerate(underground):
            #cor  = coordinate of cell in underground
            #pay = payoff of digging that cell in underground
            if pay > 0 or cor == self.len_z-1:
                hol = pay #return postive payoff for action digging this cell
            else:
                zstart=cor[-1]
                zend = cor[-1]+1
                hol=-1;
                #starts by checking cells directly bellow and neighbouring the cell expending down untill the bottom of the mine is reached or a postive holistic is reached
                while(zend < self.len_z+1 and hol < 0):
                    Slice = np.s_[zstart:zend]
                    if zend-1 == zstart:
                        values = underground[Slice] 
                        posvalue = values[values > 0]
                        if len(posvalue) > 0:
                            hol=pay+np.sum(posvalue) #check if cells below are good to dig
                        else:
                            hol=pay+np.sum(values) #checks if any combination of cells below give a postive payoff
                    else:
                        hol=pay+np.sum(underground[Slice]) #checks if any combination of cells below give a postive payoff
                    zend+=1
            self.best_actionpay[cor] = hol #append holistic into master array
        self.best_actionpay=np.append(self.top,self.best_actionpay)
    def Output_gen2d(self):
        """

        Returns
        -------
        None.
        
        Creates a holisic for a 2D mine to trim actions in BB
        creates an admistable holistic for a 3d mine, with each cell Representing an actions evental payoff
        first checks if action will result in a postive payoff in the newly dug cell, returns it as a valid action
        else checks if it neighbours directly below are postive and have a greater payoff than is lost if true it is a valid action
        else checks neigbours directly below : +1 until the max depth or a positive payoff is found
        after taking into account all losses and gains for getting to that cell, if a postive payoff is found it is a valid action
        
        valid actions have a postive value 
        values to trim have a negitive value
        
        hol: holisitc to append for an action into action array
            negitive of bad action postive if good action

        """
        self.best_actionpay = np.zeros(self.mine_shape)
        underground = np.array(self.underground)
        for cor, pay in np.ndenumerate(underground):
            #cor  = coordinate of cell in underground
            #pay = payoff of digging that cell in underground
            if pay > 0 or cor[1] == self.len_z-1:
                hol = pay #return postive payoff for action digging this cell
            else:
                zstart  =cor[1]+1
                if cor[0] == 0:
                    xstart = cor[0];
                    xend = cor[0]+2
                elif cor[0] == self.len_x-1:
                    xstart = cor[0]-1;
                    xend = cor[0]+1
                else:
                    xstart = cor[0]-1;
                    xend = cor[0]+2
                zend = zstart+1
                hol=-1;
                while(zend < self.len_z+1 and hol < 0):
                    Slice = np.s_[xstart:xend,zstart:zend]
                    if zend-1 == zstart:
                        values = underground[Slice]
                        posvalue = values[values > 0]
                        if len(posvalue) > 0:
                            hol=pay+np.sum(posvalue) #check if cells below are good to dig
                        else:
                            hol=pay+np.sum(values) #checks if any combination of cells below give a postive payoff
                    else:
                        hol=pay+np.sum(underground[Slice]) #checks if any combination of cells below give a postive payoff
                    zend+=1
            self.best_actionpay[cor] = hol #append holistic into master array
        self.best_actionpay=np.append(self.top,self.best_actionpay,axis=-1)
    

    
    def reset(self):
        """
        returns the Mine class variables back to the orignal values, so that goal_test can work
        as though no best state has been found
        """
        self.maxpayoff = -1;
        self.excavated = 0;
        
        
        # super().__init__() # call to parent class constructor not needed
        
    @functools.lru_cache(maxsize=256)
    def goal_test(self, state, payoff=None):
        '''
        

        Parameters
        ----------
        state : Tuple of current mine state

        payoff : float payoff from state, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        bool
            True if state is better than last known best state
            False if state is worse than last known best state
            
        Checks if the current state is safe and provides a greater payoff when compared against
        the current best known state 
        '''

        if None == payoff:
            payoff = self.payoff(state)
        goal_t = ((payoff > self.maxpayoff) or (payoff == self.maxpayoff and np.sum(state) < self.excavated))    
        if goal_t:
            self.excavated = np.sum(state);
            self.maxpayoff  = payoff;
        
        return goal_t;
    @functools.lru_cache(maxsize=256)
    def dependents(self,x,y=None):
        """
        

        Parameters
        ----------
        x : int, x coordinate of cell
            DESCRIPTION.
        y : int y coordinate of cell, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dependent : list of neighbouring cells
            constructs a list of neighbouring cells for coordinate of x,y

        """
        #list to store neighbours
        dependent = [];
        #checks if mine is 2D or 3D
        if y == None:
            #2D neighbours for x, 
            #add neighbour to the left if not 0
            if x>0:
                dependent.append(x-1)
            #add neighbour to the right if not edge of the array at right
            if x < self.len_x-1:
                dependent.append(x+1)
        else:
            #3D neighbours for x,y
            #add neighbour to the left if not 0
            if x > 0:
                dependent.append((x-1,y))
                #add neighbour to the left if not 0
                if y > 0:
                    dependent.append((x,y-1))
                #add neighbour to the right if not edge of the array at right
                if y<self.len_y-1:
                    dependent.append((x,y+1))
            #add neighbour to the right if not edge of the array at right
            if x < self.len_x-1:
                dependent.append((x+1,y))
                #add neighbour to the left if not 0
                if y > 0:
                    dependent.append((x,y-1))
                #add neighbour to the right if not edge of the array at right
                if y<self.len_y-1:
                    dependent.append((x,y+1))
        return dependent
    @functools.lru_cache(maxsize=256)
    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
    
    #@functools.lru_cache(maxsize=256)
    def check_dependents(self,depth,neighbors,state):
        """
        ---Helper function for actions---
        takes:
            depth = depth of cell to compare
            neighbours = array of indexs for the neighbouring cells of depth
            state = current state
        returns True if all neighbours depths >= depth         
        """
        state= np.array(state)
        depth=int(depth)

        #loops through all neighbours checking if the current depth is less than the dig tolorance 
        return np.all(np.array([True if (self.dig_tolerance>= depth+1-state[n]) and (depth< self.len_z-1) 
                                else False 
                                    for n in neighbors ]))
    @functools.lru_cache(maxsize=256)
    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''
        state = np.array(state)
        """
        returns all valid actions through cycling through every cell within state.
        for each cell within state all it's neighbours current depth is check against it's own
        and if the cell's depth < dig tolorance and not pass the mines max depth 
        it is a valid action and added to the list to return as it coordintes 
        """
        #cor  = coordinate of cell in underground
        #depth = current dug depth of the corresponding cor    
        return tuple([cor 
                      for cor, depth in np.ndenumerate(state) 
                          if self.check_dependents(depth, self.surface_neigbhours(cor),state) and (depth<self.len_z-1)]) #and (depth < self.max_dig)])
                
    @functools.lru_cache(maxsize=256)
    def result(self, state, action):
        """
        

        Parameters
        ----------
        state : tuple of the current mine state
            DESCRIPTION.
        action : tuple coordinate of cell to dig down by one

        Returns
        -------
        New_state: Tuple of child state of state with the applied action
        
        The action must be a valid actions.
        That is, one of those generated by  self.actions(state).
        """
        try:
            action = tuple(action)
        except:
            action = action
        new_state = np.array(state) # Make a copy
        new_state[action] += 1 
        new_state = convert_to_tuple(new_state)
        assert isinstance(new_state,tuple)
        return new_state
                
    
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if np.array(self.underground).ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if np.array(self.underground).ndim == 2:
            # 2D mine
            return str(np.array(self.underground).T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(np.array(self.underground)[...,z]) for z in range(self.len_z))
                    
                        
                
            return self.underground[loc[0], loc[1],:]
        
    
    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    @functools.lru_cache(maxsize=256)
    def payoff(self, state):
        '''
        

        Parameters
        ----------
        state : tuple of mine state

        Returns
        -------
        float of payoff generated from state
            the state passed into payoff is used as z c coordinate
            to index the cumsum_mine array, with the resulting sliced variables 
            being summed together to get the toal payoff of the state

        '''

        try:
            #turns state into np array for manipulation
            state =np.array(convert_to_list(state))
            #local store for mine's cummiltive sum
            cumsum_mine = np.array(convert_to_list(self.cumsum_mine))
            #reshape the state array to correspond to z values of the cumsum array
            Z_pos = np.reshape(state,-1)
            #removes any indexs that have to been dug yet
            to_remove = np.array(Z_pos > -1)
            x_pos = np.array(self.x_pos)[to_remove]
            Z_pos = Z_pos[Z_pos > -1 ]
            #returns 0 if state has not started digging yet
            if len(Z_pos)==0:
                return 0
            if self.not2D():
                y_pos = np.array(self.y_pos)[to_remove]
                #sums up all slices taken from cumsum array to produce current payoff
                return np.sum(cumsum_mine[x_pos,y_pos,Z_pos])
                
            return np.sum(cumsum_mine[x_pos,Z_pos])
        except:
            assert type(state) == tuple
            return 0


    def is_dangerous(self, state):
        '''
        

        Parameters
        ----------
        state : tuple of mine state

        Returns
        -------
        bool
            returns true if current state does not break the dig tolerance or pass 
            the mines max z length
            else returns false

        '''
        
        """
        creates a truth function where the state array is sliced into 2 corresponding arrays, 
        which line up a set of nieghbouring cells these two arrays are than minus  from each other (this is repeated for (x,), (y,) and (x,y)). 
        The resulting array is than checked to see if any number in the array is greater than the dig tolerance
        returning true if all numbers are lower or equal to the dig tolerance and false otherwise
        
        for each direction only one check is done for dig tolerance, even though each direction gives a cell 2 neighbours.
        this is done because it is not need to shift the cell back the other way due to how the slicing is done. As reversing the
        slice shifting would result in an identical dig tolerance truth array, but shited one back. this can be seen in
        nthe following example:
            state = (0,1,2,4,4,3)
            state[:-1] = (0,1,2,4,4)
            state[1:]  = (1,2,4,4,3)
            as the slices show all important neighbours are accounted for with every cell being tested 
            against their neighbours. this works because the first cell and last cell only have one neighbour meaning 
            slicing the array to include all but one row  can be done as it allows for the end cells to be tested against one neighbour
            while all other cells get tested against the other two.
        """
        state=np.array(state)
        if self.not2D:
            return np.all(np.abs(state[1:,:]-state[:-1,:] <= self.dig_tolerance)) and np.all(np.abs(state[1:,1:]-state[:-1,:-1]) <= self.dig_tolerance) and np.all((np.abs(state[:,1:]-state[:,:-1]) <= self.dig_tolerance))
        return np.all(np.abs(state[1:]-state[:-1] <= self.dig_tolerance))
    def not2D(self):
        '''

        Returns
        -------
        bool
            returns true of mine is 3D
            else returns False

        '''
        """
        checks if a given mine is 2d or 3d
        returns True if mine is 3d
        returns False if mine is 2d
        """
        if (self.len_y==0):
            return False
        return True
    @functools.lru_cache(maxsize=32)
    def path_cost(self,c=None,state=None,action=None,next_state=None):
        '''
        

        Parameters
        ----------
        c : float cost from state to next_state, optional
            DESCRIPTION. The default is None.
        state : tuple current node state, optional
            DESCRIPTION. The default is None.
        action : tuple coordinate to get from state to next_state, optional
            DESCRIPTION. The default is None.
        next_state : tuple child node state got from action, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            int 
            returns 1 if action produced a child state with a possiblity of a higher payoff
            else returns -1 if action is detremental to total payoff

        '''
        """
        assigns the cost of expending to a new node based off of the pre computed best_actionpay, 
        which creates a holistic that takes into account the payoff of a dig and what cells can be reached after digging 
        out that cell.
        assigning a postive path cost to actions that have a postive payoff 
        and a negitive to actions that have a net loss on the mine
        """
        if action == None:
            return 1
        #next state of the mine
        next_state = np.array(next_state+np.ones(self.mine_shape[:-1]),dtype = np.intp)
        #gets the coordinate of the corresponding holistic in the best action pay variable
        ind = tuple(np.append(action,next_state[action]))
        underground_above  = np.array(self.best_actionpay) 
        action_payoff = underground_above[ind]
        #returns 1 if the action is a net gain or -1 if it is a loss 
        if (action_payoff) >= 0: 
            return 1
        else:
            return -1
        
           


    
    # ========================  Class Mine  ==================================


#==== Cache class used to store max payoff and current explored states ====#
class Cache():
    """
    Cache class used in DP to store visted paths and best state
    self.cache: all states as tuples that have been visited
    self.best_payoff: best payoff found from current visited states
    self.best_state: best state found so far
    self.best_actions: list of actions to go from mine.intial to best state
    self.mine_depth: number of cells dug in best_state
    self.cached: number of states cached in self.cache
    """

    def __init__(self): 
        self.cache =[]; #states cache
        self.best_payoff=0 #best payoff
        self.best_state=None #best state
        self.best_actions = None #best action list to get to state
        self.mine_depth = 0
        self.cached=0;  #number of states cached
    def __repr__(self):
        return "<Cache {}>".format(self.cache)
    def add(self,to_cache):
        """
        Parameters
        ----------
        to_cache : tuple of mine state
            to_cache is added on to the cache list.

        Returns
        -------
        None.

        """
        if isinstance(to_cache,list):
            to_cache = convert_to_tuple(to_cache)
        assert isinstance(to_cache,tuple)
        self.cache.append(to_cache)
        self.cached+=1
    
    def check(self, to_check):
        """
        Parameters
        ----------
        to_check : tuple state of mine
            to_check is checked against cach list to check if  current state has been cached.

        Returns
        -------
        bool
            Returns true if state is in cache else false.

        """
        assert type(to_check) == tuple
        if to_check in self.cache:
            return True
        return False
    def update(self,pay,state):
        """
        Parameters
        ----------
        pay : float of mine state payoff
            pay is used to check if state is a better state than best_payoff.
        state : tuple of mine state
            new state to check.
        Updates the best payoff and best state within the cache

        Returns
        -------
        None.

        """
        state_depth =np.sum(state); #sum of how many cells have been dug
        #checks that payoff is greater than best known payoff else if both are the same choose state with less dug cells
        if (pay > self.best_payoff) or  (pay == self.best_payoff and self.mine_depth > state_depth):
            self.best_payoff = pay;
            self.best_state = state;
            self.mine_depth = state_depth;
    def cache_return(self):
        """
        

        Returns
        -------
        TYPE
            Best_payoff from cache, FLOAT
        TYPE
            best_state from cache, TUPLE
        TYPE
            best_actions list of cordintes to get to state, tuple of tuples

        """
        return self.best_payoff,self.best_actions,self.best_state

def search_dp_dig_plan(mine):
    """
    Parameters
    ----------
    mine : A Mine Class object
        DESCRIPTION.

    Returns
    -------
    best_payoff : float of best mine state total payoff
 
    best_action_list : tuple of tuples of actions to get to best_state

    best_final_state : tuple of current mine state
        
    Finds the best action by cycling through all valid action from the mines intial state
    untill no more actions can found returning the best state. 
    speed is increaded through using the cache class to store states that have already been explored  
    """
    assert isinstance(mine, Mine)
    #init cache 
    cache = Cache();
    #recursive function 
    def recurse(state):
        """
        

        Parameters
        ----------
        state : Tuple of mine state
           state is first checked against cache
           if state is in cache leave it at the current state returning
           else the new state is added to the cache and it child states are found through finding the valid actions

        Returns
        -------
        None.

        """
        assert isinstance(state, tuple)
        if cache.check(state):
            cache.update(mine.payoff(state),state)
            return None
        #check if current staet is better than best known state
        cache.update(mine.payoff(state),state)
        #update cache by adding new state
        cache.add(state)
        #check if state isn't at the bottom of the mine
        if state != mine.Max_payoff_depth:
            #generate list of actions
            for a in mine.actions(state):
                #run the recursive function for each child state of current state
                next_s = mine.result(state,a)
                #if mine.path_cost(action = a,next_state=next_s)>0:
                recurse(next_s)
        return None
    #spcial condition to return if mine is a single column
    if mine.len_x == 1:
        depth_dug = mine.initial[0]+1
        mdepth = np.argmax(mine.cumsum_mine[depth_dug:])+depth_dug
        best_final_state = tuple([mdepth,])
        best_payoff = np.max(mine.cumsum_mine)
        best_action_list = find_action_sequence(mine.initial,best_final_state)
        best_final_state = tuple([mdepth+1,])
        return best_payoff, best_action_list, best_final_state
    
    state = mine.initial #start at intital mine state
    recurse(state) #start recursive function to find best mine state 
    #find actions to get to end mine state
    cache.best_actions = convert_to_tuple(find_action_sequence(state,cache.best_state))
    cache.best_state = convert_to_tuple(cache.best_state+np.ones(np.array(cache.best_state).shape))
    #return best_payoff, best_state, best_actions
    return cache.cache_return()  

            
            
def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    """
    Parameters
    ----------
    mine : Problem class of mine to be passed to function
    

    Returns
    -------
    best_payoff : float of best mine state total payoff
 
    best_action_list : tuple of tuples of actions to get to best_state

    best_final_state : tuple of current mine state
    
    the BB function creates a graph of all valid states of the mine to find the best state, that maximises payoff 
    the BB function trims the graph by making sure each action will have a postive impact on the final state.
    This is done through a pre computed holistic that takes into account the cell to be dug and the new cells that can be reached 
    through diggin out that cell.
    """
    #spcial condition to return if mine is a single column
    if mine.len_x == 1:
        depth_dug = mine.initial[0]+1
        mdepth = np.argmax(mine.cumsum_mine[depth_dug:])+depth_dug
        best_final_state = tuple([mdepth,])
        best_payoff = np.max(mine.cumsum_mine)
        best_action_list = find_action_sequence(mine.initial,best_final_state)
        best_final_state = tuple([mdepth+1,])
        return best_payoff, best_action_list, best_final_state
    
    #first node is the intital state of the mine
    node = Node(mine.initial, path_cost = mine.path_cost(None,None,None,mine.initial))
    #First in firdt put queue, giving a breath search graph 
    frontier = FIFOQueue()
    assert isinstance(mine, Mine)
    frontier.append(node)
    explored = set() # initial empty set of explored states
    #set best node to start node
    best_node=mine.initial;
    #while valid actions exist
    while frontier:
        node = frontier.pop()
        #check if current node state is better than last best known node staet
        if mine.goal_test(node.state):
                best_node= node;
        #add to check nodes to ensure no state is checked twice 
        explored.add(node.state)
        #add new nodes to the frontier based on current nodes valid actions
            #checks that child node hasn't been expored 
            #checks that child node has a postive holistic value
        frontier.extend(child for child in node.expand(mine)
                        if child.state not in explored
                        and child not in frontier and (child.path_cost > 0))
    
    best_payoff = None
    best_action_list = None
    best_final_state = None
    try:
            best_payoff = mine.payoff(best_node.state)
            best_action_list = find_action_sequence(mine.initial, best_node.state)
            best_final_state =  convert_to_tuple(best_node.state + np.ones(np.array(best_node.state).shape))
    except:
        pass

    return best_payoff, best_action_list, best_final_state


def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1
    
    given that s0 and s1 are legal states no checking for dig tolerance needs to be done, 
    given that each level of the mine is extructed at in order before moving down to the next level
    '''  
    #makes sure both states are passed as a tuple
    if isinstance(s1,tuple):
        s1=np.array(s1)
    if isinstance(s0,tuple):
        s0=np.array(s0)
    assert isinstance(s1, np.ndarray)
    assert isinstance(s0, np.ndarray)
    actions=[];
    #keeps adding new acctions into tha action list while s1 is not s0
    while np.any(s0 != s1):
        #runs through each cell in the mine state once
        for cor, depth in np.ndenumerate(s0):
            #cor  = coordinate of cell in state s0
            #depth = current dug depth of the corresponding cor 
            #checks that the depth of a cell of s0 is less than that of the corresponding s1 cell
            if depth < s1[cor]:
                #adds one to the dug in state s0 in the current cells postion
                s0[cor]+=1;
                #add the action for digging out current cell
                actions.append(cor)
    #return actions to go from s0 to s1
    actions= convert_to_tuple(actions)
    return actions


"""--------------- Testing Functions used to tests if mine problem and alogrithms are correct ---------------"""

def mine_test_search(underground,test_name="not given", tol = 1):
    '''
    Parameters
    ----------
    underground : np.array of underground to pass into Mine class
    test_name : String used for output text to easily know what was test, optional
        DESCRIPTION. The default is "not given".
    tol : int dig tolerance to pass to Mine class, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.
    ---Ignore Function used for testing purposes---
        this function is used to construct a mine state and test both the BB and DP with the
        given underground state and dig tolerance in a black box

    '''
    #mine class
    mine = Mine(underground,tol)
    mine.reset() #check all vars are defult
    beg = time.time() #start time of BB
    #run BB function
    best_payoff, best_action_list, best_final_state = search_bb_dig_plan(mine)
    end = time.time() #end time of BB
    #printing report
    print("="*15+test_name+"="*15)
    mine.console_display()
    print("----- BB search -----")
    print("payoff:",best_payoff)
    print("final_state:", best_final_state)
    print("actions:",best_action_list)
    print('total time BB',end-beg)
    
    mine.reset()#check all vars are defult
    beg = time.time()#start time of DP
    #run DP function
    best_payoff, best_action_list, best_final_state=search_dp_dig_plan(mine)
    end = time.time() #end time of DP
    #printing report
    print("----- DP search -----")
    print("payoff:",best_payoff)
    print("final_state:",best_final_state)
    print("actions:",best_action_list)
    print('total time DP',end-beg)
    print("="*15+"end"+"="*15)
    print()
    
def precompt_2d():
    """
    Simple function for testing a 2D mine

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d = np.array([[-2,3,1,2,-4,-5],
                [5,-4,0,2,1,-2],
                [3,-1,3,-3,2,0],
                [6,4,2,-1,5,-3],
                [4,2,2,-4,0,1]])
    #run test
    mine_test_search(un2d,"pre calc 2d mine")
def digTol3():
    """
    Simple function for testing a 2D mine with a dig tolerance of 3

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d = np.array([[-2,3,1,2,-4,-5],
                [5,-4,0,2,1,-2],
                [3,-1,3,-3,2,0],
                [6,4,2,-1,5,-3],
                [4,2,2,-4,0,1]])
    #run test
    mine_test_search(un2d,"Dig Tolerance of 3",3)
def given_2d():
    """
    Simple function for testing a 2D mine

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d= np.array([
       [-0.814,  0.637, 1.824, -0.563],
       [ 0.559, -0.234, -0.366,  0.07 ],
       [ 0.175, -0.284,  0.026, -0.316],
       [ 0.212,  0.088,  0.304,  0.604],
       [-1.231, 1.558, -0.467, -0.371]])
    #run test
    mine_test_search(un2d,"sanity test 2d mine")
def given_3d():
    """
    Simple function for testing a 3D mine

    Returns
    -------
    None.

    """
    #mine underground to test
    un3d= np.array([[[ 0.455,  0.579, -0.54 , -0.995, -0.771],
                                   [ 0.049,  1.311, -0.061,  0.185, -1.959],
                                   [ 2.38 , -1.404,  1.518, -0.856,  0.658],
                                   [ 0.515, -0.236, -0.466, -1.241, -0.354]],
                                  [[ 0.801,  0.072, -2.183,  0.858, -1.504],
                                   [-0.09 , -1.191, -1.083,  0.78 , -0.763],
                                   [-1.815, -0.839,  0.457, -1.029,  0.915],
                                   [ 0.708, -0.227,  0.874,  1.563, -2.284]],
                                  [[ -0.857,  0.309, -1.623,  0.364,  0.097],
                                   [-0.876,  1.188, -0.16 ,  0.888, -0.546],
                                   [-1.936, -3.055, -0.535, -1.561, -1.992],
                                   [ 0.316,  0.97 ,  1.097,  0.234, -0.296]]])
    #run test
    mine_test_search(un3d,"sanity test32d mine")
    
def neg_dig():
    """
    Simple function for testing a mine where best state is the intial state

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d = np.full((5,5),-1)
    #run test
    mine_test_search(un2d, "don't dig test")

    
def shallow_dig():
    """
    Simple function for testing the espeed of BB and DP whe nbest state is only the top needs to be dug

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d = np.array([[1,-1,-1,-1,-1],
                 [1,1,-1,-1,-1],
                 [1,-1,-1,-1,-1],
                 [1,-1,-1,-1,-1],
                 [1,-1,-1,1,-1]])
    #run test
    mine_test_search(un2d, "shallow best state test")
    
def full_dig():
    """
    Simple function for testing the speed of a mine where best payoff is all cells dug

    Returns
    -------
    None.

    """
    #mine underground to test
    un2d = np.ones((5,5))
    #run test
    mine_test_search(un2d,"full dig best state test")
def mine_1d():
    """
    Simple function for testing a 1D mine

    Returns
    -------
    None.

    """
    #mine underground to test
    un1d = np.array([1,2,-4,2,1])
    #run test
    mine_test_search(un1d,"single column test")
    
def test_run(): 
    """
    Simple function to run all test at once

    Returns
    -------
    None.

    """
    precompt_2d();
    given_2d();
    neg_dig();
    full_dig();
    shallow_dig();
    mine_1d();
    digTol3();
    given_3d();

#function used to run tests
#test_run()