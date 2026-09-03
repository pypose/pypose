import numpy as np

class DataSplitter(object):
    '''
    shuffle the trajectories into subsets
    return the next set when queried
    if all the trajectories are emuerated, shuffle again  
    '''
    def __init__(self, trajlist, trajlenlist, framelist, framenum, shuffle=True):
        '''
        trajlist: the relative path of the trajectories [traj0, traj1, ...] 
        trajlenlist: the length of the trajectories [len0, len1, ...]
        framelist: the frames [[traj0_frame0, traj0_frame1, ...],
                               [traj1_frame0, traj1_frame1, ...],
                               ...]
        framenum: the framenum for each subset
        '''
        self.trajlist, self.trajlenlist, self.framelist = trajlist, trajlenlist, framelist
        self.framenum = framenum
        self.shuffle = shuffle
        self.trajnum = len(trajlist)
        self.totalframenum = sum(trajlenlist)

        self.trajinds = np.arange(self.trajnum, dtype=np.int32)
        self.curind = -1

    def get_next_split(self):
        framecount = 0 
        subtrajlist, subtrajlenlist, subframelist = [], [], []

        while framecount < self.framenum:
            self.curind = (self.curind + 1) % self.trajnum
            if self.curind == 0 and self.shuffle: # shuffle the trajectory 
                self.trajinds = np.random.permutation(self.trajnum)

            # add the current trajectory to the lists
            trajind = self.trajinds[self.curind]
            trajlen = self.trajlenlist[trajind]
            subtrajlist.append(self.trajlist[trajind])
            if framecount + trajlen > self.framenum: # the trajectory is too long, only add one part of it. Note that the remaining part of the trajectory will be thrown away
                addnum = self.framenum - framecount
                subtrajlenlist.append(addnum)
                subframelist.append(self.framelist[trajind][:addnum])
                framecount += addnum
            else: # add the whole trajectory to lists
                subtrajlenlist.append(trajlen)
                subframelist.append(self.framelist[trajind])
                framecount += trajlen

        return subtrajlist, subtrajlenlist, subframelist, self.framenum