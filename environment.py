import numpy as np

class Environment(object):
    def __init__(self, env_name, params, logger):
        self.CreateNetworks()
        #np.set_printoptions(threshold=np.inf)
        #self.PrintNetwork()
        adjacent = self.GetNetwork(env_name)
        self.state_size = len(adjacent)
        self.action_space = self.state_size
        self.currentState = np.zeros(self.state_size)
        self.logger = logger
        self.logger.Debug('current state', self.currentState)
        self.adjacent = adjacent
        self.slotNumber = 0
        self.node_select = params.node_select
        self.chances = params.chances
        self.remainingchances = params.chances

    def checkActionLegal(self, action):
        legal = True
        reward = 0
        if len(action) == 0:
            legal = False
            reward = -1
        for i in range(len(action)):
            for j in range(i + 1, len(action)):
                node1 = action[i]
                node2 = action[j]
                self.logger.Debug(" i ", i, " j ", j, " node1 ", node1, " node2 ", node2)
                if self.adjacent[node1][node2]  > 0:
                    legal = False
                    reward -= 1
                else:
                    reward += 1

        for node in action:
            if self.currentState[node] > 0:
                legal = False
                reward -= 1
            else:
                reward += 1

        if not legal and reward >=0:
            reward = -1

        self.logger.Debug("legal ", legal, "len(action) ", len(action), "action ", action
              , "self.currentState ", self.currentState)

        return legal, reward

    def get_action_space(self):
        self.logger.Debug("self.action_space", self.action_space)
        return self.action_space

    def get_state_size(self):
        self.logger.Debug("self.state_size", self.state_size)
        return self.state_size

    def get_state_shape(self):
        self.logger.Debug("self.currentState.shape", self.currentState.shape)
        return self.currentState.shape

    def get_num_action(self):
        self.logger.Debug("self.currentState.shape[0]", self.currentState.shape[0])
        return self.currentState.shape[0]

    def reset(self):
        self.slotNumber = 0
        self.currentState = np.zeros(self.state_size)
        self.remainingchances = self.chances
        return self.currentState

    def isDone(self):
        if self.remainingchances == 0 :
            return True

        for i in range(self.state_size):
            if self.currentState[i] == 0:
                return False
        return True

    def convertAction(self, action):
        nodeList = []
        index = 0
        for node in action:
            self.logger.Debug('node value', node)
            if node >= self.node_select:
                nodeList.append(index)
            index += 1
        return nodeList

    def getSchedule(self):
        return self.currentState

    def reward(self, nodeList):
        reward = len(nodeList) #/ self.state_size
        return reward

    def step(self,action):
        nodeList = self.convertAction(action)
        legal, reward = self.checkActionLegal(nodeList)
        self.logger.Debug(" check legal 1 ", legal)
        if legal:
            self.logger.Debug(" check legal 2 ", legal)
            self.slotNumber += 1
            for node in nodeList:
                self.currentState[node] = self.slotNumber
            reward += self.reward(nodeList)
        else :
            self.remainingchances -= 1
        done = self.isDone()
        self.logger.Debug(" nodeList len ", len(nodeList), " legal ", legal, " node list "
              , nodeList, " slot number ", self.slotNumber, " current State "
              , self.currentState, " reward ", reward)
        next_state = self.currentState
        info = {}
        rewardarray = np.zeros(1)
        rewardarray[0] = reward
        return next_state, rewardarray, done, info

    def GetNetwork(self, name):
        if(name == "network_1"):
            adjacent = self.adj1
        elif(name == "network_2"):
            adjacent = self.adj2
        elif(name == "network_3"):
            adjacent = self.adj3
        elif(name == "network_4"):
            adjacent = self.adj4
        else:
            adjacent = self.adj5
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        return adjacent

    def PrintNetwork(self):
        adjacent = self.adj1
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        print(adjacent)
        adjacent = self.adj2
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        print(adjacent)
        adjacent = self.adj3
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        print(adjacent)
        adjacent = self.adj4
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        print(adjacent)
        adjacent = self.adj5
        for i in range(adjacent.shape[0]):
            for j in range(adjacent.shape[1]):
                if adjacent[i][j] > 0:
                    adjacent[j][i] = adjacent[i][j]
        print(adjacent)

    def CreateNetworks(self):
        #grid topology 1
        # 0  1  2
        # 3  4  5
        # 6  7  8

        #optimal schedule
        #slot 0   0
        #slot 1   1
        #slot 2   2
        #slot 3   3
        #slot 4   4
        #slot 5   5
        #slot 6   6
        #slot 7   7
        #slot 8   8

        adj1 = np.zeros((9,9))
        adj1[0][1], adj1[0][3] = 1, 1
        adj1[1][2], adj1[1][4] = 1, 1
        adj1[2][5] = 1
        adj1[3][4], adj1[3][6] = 1, 1
        adj1[4][5], adj1[4][7] = 1, 1
        adj1[5][8] = 1
        adj1[6][7] = 1

        adj1[0][2], adj1[0][4], adj1[0][6] = 2, 2, 2
        adj1[1][3], adj1[1][5], adj1[1][7] = 2, 2, 2
        adj1[2][4], adj1[2][8] = 2, 2
        adj1[3][5], adj1[3][7] = 2, 2
        adj1[4][6], adj1[4][8] = 2, 2
        adj1[5][7] = 2
        adj1[6][8] = 2

        self.adj1 = adj1

        #grid topology 2
        # 0   1   2   3
        # 4   5   6   7
        # 8   9   10  11
        # 12  13  14  15

        #optimal schedule
        #slot 0   0, 3, 12, 15
        #slot 1   1, 13
        #slot 2   2, 14
        #slot 3   4, 7
        #slot 4   5
        #slot 5   6
        #slot 6   8, 11
        #slot 7   9
        #slot 8   10

        adj2 = np.zeros((16,16))
        adj2[0][1], adj2[0][4] = 1, 1
        adj2[1][2], adj2[1][5] = 1, 1
        adj2[2][3], adj2[2][6] = 1, 1
        adj2[3][7] = 1
        adj2[4][5], adj2[4][8] = 1, 1
        adj2[5][6], adj2[5][9] = 1, 1
        adj2[6][7], adj2[6][10] = 1, 1
        adj2[7][11] = 1
        adj2[8][9], adj2[8][12] = 1, 1
        adj2[9][10], adj2[9][13] = 1, 1
        adj2[10][11], adj2[10][14] = 1, 1
        adj2[11][15] = 1
        adj2[12][13] = 1
        adj2[13][14] = 1
        adj2[14][15] = 1

        adj2[0][2], adj2[0][5], adj2[0][8] = 2, 2, 2
        adj2[1][3], adj2[1][4], adj2[1][6], adj2[1][9] = 2, 2, 2, 2
        adj2[2][5], adj2[2][7], adj2[2][10] = 2, 2, 2
        adj2[3][6], adj2[3][11] = 2, 2
        adj2[4][6], adj2[4][9], adj2[4][12] = 2, 2, 2
        adj2[5][7], adj2[5][8], adj2[5][10], adj2[5][13] = 2, 2, 2, 2
        adj2[6][9], adj2[6][11], adj2[6][14] = 2, 2, 2
        adj2[7][10], adj2[7][15] = 2, 2
        adj2[8][11], adj2[8][13] = 2, 2
        adj2[9][11], adj2[9][12], adj2[9][14] = 2, 2, 2
        adj2[10][13], adj2[10][15] = 2, 2
        adj2[11][14] = 2
        adj2[12][14] = 2

        self.adj2 = adj2

        #grid topology 3
        # 0   1   2   3   4
        # 5   6   7   8   9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # 20 21 22 23 24

        #optimal schedule
        #slot 0   0, 3, 15, 18
        #slot 1   1, 4, 16, 19
        #slot 2   2, 17
        #slot 3   5, 8, 20, 23
        #slot 4   6, 9, 21, 24
        #slot 5   7, 22
        #slot 6   10, 13
        #slot 7   11, 14
        #slot 8   12

        adj3 = np.zeros((25,25))
        adj3[0][1], adj3[0][5] = 1, 1
        adj3[1][2], adj3[1][6] = 1, 1
        adj3[2][3], adj3[2][7] = 1, 1
        adj3[3][4], adj3[3][8] = 1, 1
        adj3[4][9] = 1
        adj3[5][6], adj3[5][10] = 1, 1
        adj3[6][7], adj3[6][11] = 1, 1
        adj3[7][8], adj3[7][12] = 1, 1
        adj3[8][9], adj3[8][13] = 1, 1
        adj3[9][14] = 1
        adj3[10][11], adj3[10][15] = 1, 1
        adj3[11][12], adj3[11][16] = 1, 1
        adj3[12][13], adj3[12][17] = 1, 1
        adj3[13][14], adj3[13][18] = 1, 1
        adj3[14][18] = 1
        adj3[15][16], adj3[15][20] = 1, 1
        adj3[16][17], adj3[16][21] = 1, 1
        adj3[17][18], adj3[17][22] = 1, 1
        adj3[18][19], adj3[18][23] = 1, 1
        adj3[19][24] = 1
        adj3[20][21] = 1
        adj3[21][22] = 1
        adj3[22][23] = 1
        adj3[23][24] = 1

        adj3[0][2], adj3[0][6], adj3[0][10] = 2, 2, 2
        adj3[1][3], adj3[1][5], adj3[1][7], adj3[1][11] = 2, 2, 2, 2
        adj3[2][4], adj3[2][6], adj3[2][8], adj3[2][12] = 2, 2, 2, 2
        adj3[3][7], adj3[3][9], adj3[3][13] = 2, 2, 2
        adj3[4][8], adj3[4][14] = 2, 2
        adj3[5][7], adj3[5][11], adj3[5][15] = 2, 2, 2
        adj3[6][8], adj3[6][12], adj3[6][16] = 2, 2, 2
        adj3[7][9], adj3[7][11], adj3[7][13], adj3[7][17] = 2, 2, 2, 2
        adj3[8][12], adj3[8][14], adj3[8][18] = 2, 2, 2
        adj3[9][13], adj3[9][19] = 2, 2
        adj3[10][12], adj3[10][16], adj3[10][20] = 2, 2, 2
        adj3[11][13], adj3[11][15], adj3[11][17], adj3[11][21] = 2, 2, 2, 2
        adj3[12][14], adj3[12][16], adj3[12][18], adj3[12][22] = 2, 2, 2, 2
        adj3[13][17], adj3[13][19], adj3[13][23] = 2, 2, 2
        adj3[14][18], adj3[14][24] = 2, 2
        adj3[15][17], adj3[15][21] = 2, 2
        adj3[16][18], adj3[16][20], adj3[16][22] = 2, 2, 2
        adj3[17][19], adj3[17][21], adj3[17][23]= 2, 2, 2
        adj3[18][22], adj3[18][24] = 2, 2
        adj3[19][23] = 1
        adj3[20][22] = 1
        adj3[21][23] = 1
        adj3[22][24] = 1


        self.adj3 = adj3

        #grid topology 4
        # 0   1   2   3   4   5
        # 6   7   8   9   10 11
        # 12 13 14 15 16 17
        # 18 19 20 21 22 23
        # 24 25 26 27 28 29
        # 30 31 32 33 34 35

        #optimal schedule
        #slot 0   0, 3, 18, 21
        #slot 1   1, 4, 19, 22
        #slot 2   2, 5, 20, 23
        #slot 3   6, 9, 24, 27
        #slot 4   7, 10, 25, 28
        #slot 5   8, 11, 26, 29
        #slot 6   12, 15, 30, 33
        #slot 7   13, 16, 31, 34
        #slot 8   14, 17, 32, 35

        adj4 = np.zeros((36,36))
        adj4[0][1], adj4[0][6] = 1, 1
        adj4[1][2], adj4[1][7] = 1, 1
        adj4[2][3], adj4[2][8] = 1, 1
        adj4[3][4], adj4[3][9] = 1, 1
        adj4[4][5], adj4[4][10] = 1, 1
        adj4[5][11] = 1
        adj4[6][7], adj4[6][12] = 1, 1
        adj4[7][8], adj4[7][13] = 1, 1
        adj4[8][9], adj4[8][14] = 1, 1
        adj4[9][10], adj4[9][15] = 1, 1
        adj4[10][11], adj4[10][16] = 1, 1
        adj4[11][17] = 1
        adj4[12][13], adj4[12][18] = 1, 1
        adj4[13][14], adj4[13][19] = 1, 1
        adj4[14][15], adj4[14][26] = 1, 1
        adj4[15][16], adj4[15][21] = 1, 1
        adj4[16][17], adj4[16][22] = 1, 1
        adj4[17][23] = 1
        adj4[18][19], adj4[18][24] = 1, 1
        adj4[19][20], adj4[19][25] = 1, 1
        adj4[20][21], adj4[20][26] = 1, 1
        adj4[21][22], adj4[21][27] = 1, 1
        adj4[22][23], adj4[22][28] = 1, 1
        adj4[23][29] = 1
        adj4[24][25], adj4[24][30] = 1, 1
        adj4[25][26], adj4[25][31] = 1, 1
        adj4[26][27], adj4[26][32] = 1, 1
        adj4[27][28], adj4[27][33] = 1, 1
        adj4[28][29], adj4[28][34] = 1, 1
        adj4[29][35] = 1
        adj4[30][31] = 1
        adj4[31][32] = 1
        adj4[32][33] = 1
        adj4[33][34] = 1
        adj4[34][35] = 1

        adj4[0][2], adj4[0][7], adj4[0][12] = 2, 2, 2
        adj4[1][3], adj4[1][6], adj4[1][8], adj4[1][13] = 2, 2, 2, 2
        adj4[2][4], adj4[2][7], adj4[2][9], adj4[2][14] = 2, 2, 2, 2
        adj4[3][5], adj4[3][8], adj4[3][10], adj4[3][15] = 2, 2, 2, 2
        adj4[4][9], adj4[4][11], adj4[4][16] = 2, 2, 2
        adj4[5][10], adj4[5][7] = 2, 2
        adj4[6][8], adj4[6][13], adj4[6][18] = 2, 2, 2
        adj4[7][9], adj4[7][12], adj4[7][14], adj4[7][19] = 2, 2, 2, 2
        adj4[8][10], adj4[8][13], adj4[8][15], adj4[8][20] = 2, 2, 2, 2
        adj4[9][11], adj4[9][14], adj4[9][16], adj4[9][21] = 2, 2, 2, 2
        adj4[10][15], adj4[10][17], adj4[10][22] = 2, 2, 2
        adj4[11][16], adj4[11][23] = 2, 2
        adj4[12][14], adj4[12][19], adj4[12][24] = 2, 2, 2
        adj4[13][15], adj4[13][18], adj4[13][20], adj4[13][25] = 2, 2, 2, 2
        adj4[14][16], adj4[14][19], adj4[14][21], adj4[14][26] = 2, 2, 2, 2
        adj4[15][17], adj4[15][20], adj4[15][22], adj4[15][27] = 2, 2, 2, 2
        adj4[16][21], adj4[16][23], adj4[16][28] = 2, 2, 2
        adj4[17][22], adj4[17][29] = 2, 2
        adj4[18][20], adj4[18][25], adj4[18][30] = 2, 2, 2
        adj4[19][21], adj4[19][24], adj4[19][26], adj4[19][31] = 2, 2, 2, 2
        adj4[20][22], adj4[20][25], adj4[20][27], adj4[20][32] = 2, 2, 2, 2
        adj4[21][23], adj4[21][26], adj4[21][28], adj4[21][28] = 2, 2, 2, 2
        adj4[22][27], adj4[22][29], adj4[22][34] = 2, 2, 2
        adj4[23][28], adj4[23][35] = 2, 2
        adj4[24][26], adj4[24][31] = 2, 2
        adj4[25][27], adj4[25][30], adj4[25][32] = 2, 2, 2
        adj4[26][28], adj4[26][31], adj4[26][33] = 2, 2, 2
        adj4[27][29], adj4[27][32], adj4[27][34] = 2, 2, 2
        adj4[28][33], adj4[28][35] = 2, 2
        adj4[29][34] = 2
        adj4[30][32] = 2
        adj4[31][33] = 2
        adj4[32][34] = 2
        adj4[33][35] = 2

        self.adj4 = adj4

        #grid topology 5
        # 0    1   2   3   4   5   6
        # 7    8   9  10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31 32 33 34
        # 35 36 37 38 39 40 41
        # 42 43 44 45 46 47 48

        #optimal schedule
        #slot 0   0, 3, 6, 21, 24, 27, 42, 45, 48
        #slot 1   1, 4, 22, 25, 43, 46
        #slot 2   2, 5, 23, 26, 44, 47
        #slot 3   7, 10, 13, 28, 31, 41
        #slot 4   8, 11, 20, 29, 32
        #slot 5   9, 12, 30, 33
        #slot 6   14, 17, 34, 35, 38
        #slot 7   15, 18, 36, 39
        #slot 8   16, 19, 37, 40

        adj5 = np.zeros((49,49))
        adj5[0][1], adj5[0][7] = 1, 1
        adj5[1][2], adj5[1][8] = 1, 1
        adj5[2][3], adj5[2][9] = 1, 1
        adj5[3][4], adj5[3][10] = 1, 1
        adj5[4][5], adj5[4][11] = 1, 1
        adj5[5][6], adj5[5][12] = 1, 1
        adj5[6][13] = 1
        adj5[7][8], adj5[7][14] = 1, 1
        adj5[8][9], adj5[8][15] = 1, 1
        adj5[9][10], adj5[9][16] = 1, 1
        adj5[10][11], adj5[10][17] = 1, 1
        adj5[11][12], adj5[11][18] = 1, 1
        adj5[12][13], adj5[12][19] = 1, 1
        adj5[13][20] = 1
        adj5[14][15], adj5[14][21] = 1, 1
        adj5[15][16], adj5[15][22] = 1, 1
        adj5[16][17], adj5[16][23] = 1, 1
        adj5[17][18], adj5[17][24] = 1, 1
        adj5[18][19], adj5[18][25] = 1, 1
        adj5[19][20], adj5[19][26] = 1, 1
        adj5[20][27] = 1
        adj5[21][22], adj5[21][28] = 1, 1
        adj5[22][23], adj5[22][29] = 1, 1
        adj5[23][24], adj5[23][30] = 1, 1
        adj5[24][25], adj5[24][31] = 1, 1
        adj5[25][26], adj5[25][32] = 1, 1
        adj5[26][27], adj5[26][33] = 1, 1
        adj5[27][34] = 1
        adj5[28][29], adj5[28][35] = 1, 1
        adj5[29][30], adj5[29][36] = 1, 1
        adj5[30][31], adj5[30][37] = 1, 1
        adj5[31][32], adj5[31][38] = 1, 1
        adj5[32][33], adj5[32][39] = 1, 1
        adj5[33][34], adj5[33][40] = 1, 1
        adj5[34][41] = 1
        adj5[35][36], adj5[35][42] = 1, 1
        adj5[36][37], adj5[36][43] = 1, 1
        adj5[37][38], adj5[37][44] = 1, 1
        adj5[38][39], adj5[38][45] = 1, 1
        adj5[39][40], adj5[39][40] = 1, 1
        adj5[40][41], adj5[40][47] = 1, 1
        adj5[41][48] = 1
        adj5[42][43] = 1
        adj5[43][43] = 1
        adj5[44][45] = 1
        adj5[45][46] = 1
        adj5[46][47] = 1
        adj5[47][48] = 1

        adj5[0][2], adj5[0][8], adj5[0][14] = 2, 2, 2
        adj5[1][3], adj5[1][7], adj5[1][9], adj5[1][15] = 2, 2, 2, 2
        adj5[2][4], adj5[2][8], adj5[2][10], adj5[2][16] = 2, 2, 2, 2
        adj5[3][5], adj5[3][9], adj5[3][11], adj5[3][17] = 2, 2, 2, 2
        adj5[4][6], adj5[4][10], adj5[4][12], adj5[4][18] = 2, 2, 2, 2
        adj5[5][11], adj5[5][13], adj5[5][19] = 2, 2, 2
        adj5[6][12], adj5[6][20] = 2, 2
        adj5[7][9], adj5[7][15], adj5[7][21] = 2, 2, 2
        adj5[8][10], adj5[8][14], adj5[8][16], adj5[8][22] = 2, 2, 2, 2
        adj5[9][11], adj5[9][15], adj5[9][17], adj5[9][23] = 2, 2, 2, 2
        adj5[10][12], adj5[10][16], adj5[10][18], adj5[10][23] = 2, 2, 2, 2
        adj5[11][13], adj5[11][17], adj5[11][19], adj5[11][25] = 2, 2, 2, 2
        adj5[12][18], adj5[12][20], adj5[12][26] = 2, 2, 2
        adj5[13][19], adj5[13][27] = 2, 2
        adj5[14][16], adj5[14][22], adj5[14][28] = 2, 2, 2
        adj5[15][17], adj5[15][21], adj5[15][23], adj5[15][29] = 2, 2, 2, 2
        adj5[16][18], adj5[16][22], adj5[16][24], adj5[16][30] = 2, 2, 2, 2
        adj5[17][19], adj5[17][23], adj5[17][25], adj5[17][31] = 2, 2, 2, 2
        adj5[18][20], adj5[18][24], adj5[18][26], adj5[18][32] = 2, 2, 2, 2
        adj5[19][25], adj5[19][27], adj5[19][33] = 2, 2, 2
        adj5[20][26], adj5[20][34] = 2, 2
        adj5[21][23], adj5[21][29], adj5[21][35] = 2, 2, 2
        adj5[22][24], adj5[22][28], adj5[22][30], adj5[22][36] = 2, 2, 2, 2
        adj5[23][25], adj5[23][29], adj5[23][31], adj5[23][37] = 2, 2, 2, 2
        adj5[24][26], adj5[24][30], adj5[24][32], adj5[24][38] = 2, 2, 2, 2
        adj5[25][27], adj5[25][31], adj5[25][33], adj5[25][39] = 2, 2, 2, 2
        adj5[26][32], adj5[26][34], adj5[26][40] = 2, 2, 2
        adj5[27][33], adj5[27][41] = 2, 2
        adj5[28][30], adj5[28][36], adj5[28][42] = 2, 2, 2
        adj5[29][31], adj5[29][35], adj5[29][37], adj5[29][43] = 2, 2, 2, 2
        adj5[30][32], adj5[30][36], adj5[30][38], adj5[30][44] = 2, 2, 2, 2
        adj5[31][33], adj5[31][37], adj5[31][39], adj5[31][45] = 2, 2, 2, 2
        adj5[32][34], adj5[32][38], adj5[32][40], adj5[32][46] = 2, 2, 2, 2
        adj5[33][39], adj5[33][41], adj5[33][47] = 2, 2, 2
        adj5[34][40], adj5[34][48] = 2, 2
        adj5[35][37], adj5[35][43] = 2, 2
        adj5[36][38], adj5[36][42], adj5[36][44] = 2, 2, 2
        adj5[37][39], adj5[37][43], adj5[37][45] = 2, 2, 2
        adj5[38][40], adj5[38][44], adj5[38][46] = 2, 2, 2
        adj5[39][41], adj5[39][45], adj5[39][47] = 2, 2, 2
        adj5[40][46], adj5[40][48] = 2, 2
        adj5[41][47] = 2
        adj5[42][44] = 2
        adj5[43][45] = 2
        adj5[44][46] = 2
        adj5[45][47] = 2
        adj5[46][48] = 2

        self.adj5 = adj5

        #print("adj1")
        #print(adj1)
        #print("adj2")
        #print(adj2)
        #print("adj3")
        #print(adj3)
        #print("adj4")
        #print(adj4)
        #print("adj5")
        #print(adj5)


class Solution():
    def __init__(self):
        self.MaxNonZero = 0
        self.MaxSlot = 0

    def update(self, maxnonzero, maxslot):
        self.MaxNonZero = maxnonzero
        self.MaxSlot = maxslot

    def get(self):
        return self.MaxNonZero, self.MaxSlot
