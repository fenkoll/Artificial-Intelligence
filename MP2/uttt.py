from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=4 # Intened to be 4, 3 will lead to wrong answer
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #self.startBoardIdx=randint(0,8) # We do not use this code. Instead we define random startBoardIdx later in playGameYourAgent and playGameHuman

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True
        self.point = (0,0)
        self.count = 0

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        winner=0
        for i in self.globalIdx:
            for j in range(3):
                if(self.board[i[0]+j][i[1]]+self.board[i[0]+j][i[1]+1]+self.board[i[0]+j][i[1]+2]=="XXX"):
                    winner = 1
                    return winner
                elif(self.board[i[0]+j][i[1]]+self.board[i[0]+j][i[1]+1]+self.board[i[0]+j][i[1]+2]=="OOO"):
                    winner = -1
                    return winner
                elif(self.board[i[0]][i[1]+j]+self.board[i[0]+1][i[1]+j]+self.board[i[0]+2][i[1]+j]=="XXX"):
                    winner = 1
                    return winner
                elif(self.board[i[0]][i[1]+j]+self.board[i[0]+1][i[1]+j]+self.board[i[0]+2][i[1]+j]=="OOO"):
                    winner = -1
                    return winner
            if(self.board[i[0]][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]+2][i[1]+2]=="XXX" or self.board[i[0]+2][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]][i[1]+2]=="XXX"):
                winner = 1
                return winner
            elif(self.board[i[0]][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]+2][i[1]+2]=="OOO" or self.board[i[0]+2][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]][i[1]+2]=="OOO"):
                winner = -1
                return winner
        return winner

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        flag = False

        if isMax==True:
            #print("checkpt1")
            #print("self.checkWinner() is ",self.checkWinner())
             if self.checkWinner() == 1:
                score = self.winnerMaxUtility
                return score

             else:
                for i in self.globalIdx:
                    for j in range(3):
                        result_row = self.board[i[0]+j][i[1]]+self.board[i[0]+j][i[1]+1]+self.board[i[0]+j][i[1]+2]
                        result_col = self.board[i[0]][i[1]+j]+self.board[i[0]+1][i[1]+j]+self.board[i[0]+2][i[1]+j]
                        if(result_row=="_XX"or result_row == "X_X"or result_row=="XX_"):
                            score += self.twoInARowMaxUtility
                            flag = True

                        elif(result_row=="OOX"or result_row == "XOO"or result_row=="OXO"):
                            score += self.preventThreeInARowMaxUtility
                            flag = True

                        if(result_col=="_XX"or result_col=="X_X"or result_col=="XX_"):
                            score += self.twoInARowMaxUtility
                            flag = True
                        elif(result_col=="OOX"or result_col == "XOO"or result_col=="OXO"):
                            score += self.preventThreeInARowMaxUtility
                            flag = True

                    result_dia1 = self.board[i[0]][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]+2][i[1]+2]
                    result_dia2 = self.board[i[0]+2][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]][i[1]+2]
                    if(result_dia1=="_XX"or result_dia1=="X_X"or result_dia1=="XX_"):
                        score+=self.twoInARowMaxUtility
                        flag = True
                    elif(result_dia1=="OOX"or result_dia1 == "XOO" or result_dia1 =="OXO")   :
                        score += self.preventThreeInARowMaxUtility
                        flag = True

                    if(result_dia2=="_XX"or result_dia2=="X_X"or result_dia2=="XX_"):
                        score+=self.twoInARowMaxUtility
                        flag = True
                    elif(result_dia2=="OOX"or result_dia2 == "XOO"or result_dia2=="OXO")   :
                        score += self.preventThreeInARowMaxUtility
                        flag = True
                if flag==True:
                    return score

                for i in self.globalIdx:
                        if self.board[i[0]][i[1]] == "X":
                            score += self.cornerMaxUtility
                        if self.board[i[0]+2][i[1]] == "X":
                            score += self.cornerMaxUtility
                        if self.board[i[0]][i[1]+2] == "X":
                            score += self.cornerMaxUtility
                        if self.board[i[0]+2][i[1]+2] == "X":
                            score += self.cornerMaxUtility
                return score
            
            
        else:
             if self.checkWinner() == -1:
                score = self.winnerMinUtility
                return score


             else:
                    for i in self.globalIdx:
                        for j in range(3):
                            result_row = self.board[i[0]+j][i[1]]+self.board[i[0]+j][i[1]+1]+self.board[i[0]+j][i[1]+2]
                            result_col = self.board[i[0]][i[1]+j]+self.board[i[0]+1][i[1]+j]+self.board[i[0]+2][i[1]+j]

                            if(result_row=="OXX"or result_row == "XOX"or result_row=="XXO"):
                                score += self.preventThreeInARowMinUtility
                                flag = True

                            elif(result_row=="OO_"or result_row == "_OO"or result_row=="O_O"):
                                score += self.twoInARowMinUtility
                                flag = True

                            if(result_col=="OXX"or result_col=="XOX"or result_col=="XXO"):
                                score += self.preventThreeInARowMinUtility
                                flag = True
                            elif(result_col=="OO_"or result_col == "_OO"or result_col=="O_O"):
                                score += self.twoInARowMinUtility
                                flag = True
                        result_dia1 = self.board[i[0]][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]+2][i[1]+2]
                        result_dia2 = self.board[i[0]+2][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]][i[1]+2]

                        if(result_dia1=="OXX"or result_dia1=="XOX" or result_dia1=="XXO"):
                            score+=self.preventThreeInARowMinUtility
                            flag = True
                        elif(result_dia1=="OO_"or result_dia1 == "_OO"or result_dia1 =="O_O")   :
                            score += self.twoInARowMinUtility
                            flag = True

                        if(result_dia2=="OXX"or result_dia2=="XOX"or result_dia2=="XXO"):
                            score +=self.preventThreeInARowMinUtility
                            flag = True
                        elif(result_dia2=="OO_"or result_dia2 == "_OO"or result_dia2=="O_O")   :
                            score += self.twoInARowMinUtility
                            flag = True
                    if flag==True:
                        return score

                    for i in self.globalIdx:
                            if self.board[i[0]][i[1]] == "O":
                                score += self.cornerMinUtility
                            if self.board[i[0]+2][i[1]] == "O":
                                score += self.cornerMinUtility
                            if self.board[i[0]][i[1]+2] == "O":
                                score += self.cornerMinUtility
                            if self.board[i[0]+2][i[1]+2] == "O":
                                score += self.cornerMinUtility
                    return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        flag = False

        # if isMax == True:

        if self.checkWinner() == 1:
                    score = 10000
                    return score
        elif self.checkWinner() == -1:
                    score = -10000
                    return score
        else:
                for i in self.globalIdx:
                    for j in range(3):
                        result_row = self.board[i[0]+j][i[1]]+self.board[i[0]+j][i[1]+1]+self.board[i[0]+j][i[1]+2]
                        result_col = self.board[i[0]][i[1]+j]+self.board[i[0]+1][i[1]+j]+self.board[i[0]+2][i[1]+j]
                        if(result_row=="_XX"or result_row == "X_X"or result_row=="XX_"):
                            score += 500
                            flag = True
                        elif(result_row=="OXX"or result_row == "XOX"or result_row=="XXO"):
                            score -= 500
                            flag = True
                        elif(result_row=="OOX"or result_row == "XOO"or result_row=="OXO"):
                            score += 100
                            flag = True
                        elif(result_row=="OO_"or result_row == "_OO"or result_row=="O_O"):
                            score -= 100
                            flag = True
                        if(result_col=="_XX"or result_col=="X_X"or result_col=="XX_"):
                            score += 500
                            flag = True
                        elif(result_col=="OOX"or result_col == "XOO"or result_col=="OXO"):
                            score += 100
                            flag = True
                        elif(result_col=="OXX"or result_col=="XOX"or result_col=="XXO"):
                            score -= 500
                            flag = True
                        elif(result_col=="OO_"or result_col == "_OO"or result_col=="O_O"):
                            score -= 100
                            flag = True
                    result_dia1 = self.board[i[0]][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]+2][i[1]+2]
                    result_dia2 = self.board[i[0]+2][i[1]]+self.board[i[0]+1][i[1]+1]+self.board[i[0]][i[1]+2]
                    if(result_dia1=="_XX"or result_dia1=="X_X"or result_dia1=="XX_"):
                        score+=500
                        flag = True
                    elif(result_dia1=="OOX"or result_dia1 == "XOO" or result_dia1 =="OXO")   :
                        score += 100
                        flag = True
                    elif(result_dia1=="OXX"or result_dia1=="XOX" or result_dia1=="XXO"):
                        score-=500
                        flag = True
                    elif(result_dia1=="OO_"or result_dia1 == "_OO"or result_dia1 =="O_O")   :
                        score -= 100
                        flag = True
                    if(result_dia2=="_XX"or result_dia2=="X_X"or result_dia2=="XX_"):
                        score+=500
                        flag = True
                    elif(result_dia2=="OOX"or result_dia2 == "XOO"or result_dia2=="OXO")   :
                        score += 100
                        flag = True
                    elif(result_dia2=="OXX"or result_dia2=="XOX"or result_dia2=="XXO"):
                        score-=500
                        flag = True
                    elif(result_dia2=="OO_"or result_dia2 == "_OO"or result_dia2=="O_O")   :
                        score -= 100
                        flag = True
                if flag==True:
                    return score
        for i in self.globalIdx:
                if self.board[i[0]][i[1]] == "X":
                    score += 30
                if self.board[i[0]+2][i[1]] == "X":
                    score += 30
                if self.board[i[0]][i[1]+2] == "X":
                    score += 30
                if self.board[i[0]+2][i[1]+2] == "X":
                    score += 30
                if self.board[i[0]][i[1]] == "O":
                    score -= 30
                if self.board[i[0]+2][i[1]] == "O":
                    score -= 30
                if self.board[i[0]][i[1]+2] == "O":
                    score -= 30
                if self.board[i[0]+2][i[1]+2] == "O":
                    score -= 30
        return score
    

    def checkMovesLeft(self,currBoardIdx):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        
        movesLeft=False
        i = self.globalIdx[currBoardIdx]
        for j in range(3):
            for k in range(3):
                if self.board[i[0]+j][i[1]+k] == "_":
                    movesLeft = True
        return movesLeft



    def checkMoveList(self,currBoardIdx):
        moveList = []
        # print(currBoardIdx)
        i = self.globalIdx[currBoardIdx]
        for j in range(3):
            for k in range(3):
                if (self.board[i[0]+j][i[1]+k]=="_"):
                    moveList.append((i[0]+j,i[1]+k))
        # moveSequence.append(moveList)
        return moveList


    def minimax(self, depth, currBoardIdx, isMax): # minimax is used for the predefinied agent
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        point = (0,0)

        preBoardIdx = 0

        if (depth==self.maxDepth):
            self.count += 1
            return self.evaluatePredifined(isMax)
        if (isMax == True):
            if (self.checkWinner()==1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            if (self.checkWinner()==-1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            bestValue = -inf
            moveSequence = self.checkMoveList(currBoardIdx)
            for i in moveSequence:
                self.board[i[0]][i[1]] = "X"
                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == self.maxDepth:
                    currBoardIdx = preBoardIdx
                value = self.minimax(depth+1,currBoardIdx,False)
                currBoardIdx = preBoardIdx
                if value>bestValue:
                    bestValue = value
                    if depth == 0:
                        self.point = i               
                self.board[i[0]][i[1]] = "_"

        elif(isMax == False):

            if (self.checkWinner()==-1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax) ###check later

            elif (self.checkWinner()==1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            bestValue = inf
            moveSequence = self.checkMoveList(currBoardIdx)

            for i in moveSequence:

                self.board[i[0]][i[1]] = "O"
                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == self.maxDepth:
                    currBoardIdx = preBoardIdx

                value = self.minimax(depth+1,currBoardIdx,True)

                currBoardIdx = preBoardIdx

                if bestValue>value:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                self.board[i[0]][i[1]] = "_"
        return bestValue



    def minimax2(self, depth, currBoardIdx, isMax):  # minimax2 is used for my designed agent
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        point = (0,0)
        preBoardIdx = 0
        if (self.checkWinner()==1 or self.checkWinner()==-1):
            return self.evaluateDesigned(isMax)
        if (depth==4):

            self.count += 1
            return self.evaluateDesigned(isMax)
        if (isMax == True):
            bestValue = -10000000
            moveSequence = self.checkMoveList(currBoardIdx)
            #print(moveSequence)
            for i in moveSequence:

                self.board[i[0]][i[1]] = "X"

                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == 4:
                    currBoardIdx = preBoardIdx

                value = self.minimax2(depth+1,currBoardIdx,False)

                currBoardIdx = preBoardIdx
                if value>bestValue:
                    bestValue = value
                    if depth == 0:
                        self.point = i  
                self.board[i[0]][i[1]] = "_"

        elif(isMax == False):
            bestValue = 100000000
            moveSequence = self.checkMoveList(currBoardIdx)

            for i in moveSequence:
                self.board[i[0]][i[1]] = "O"
                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == 4:
                    currBoardIdx = preBoardIdx
                value = self.minimax2(depth+1,currBoardIdx,True)
                currBoardIdx = preBoardIdx

                if bestValue>value:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                
                self.board[i[0]][i[1]] = "_"

        return bestValue

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax): # alphabeta is used for the predefinied agent
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        bestValue=0.0
        preBoardIdx = 0


        if (depth==self.maxDepth):
            self.count += 1
            return self.evaluatePredifined(isMax)
        if (isMax == True):
            if (self.checkWinner()==1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            if (self.checkWinner()==-1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            bestValue = -100000000
            moveList = self.checkMoveList(currBoardIdx)
            for i in moveList:
                self.board[i[0]][i[1]] = "X"

                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == self.maxDepth:
                    currBoardIdx = preBoardIdx

                value = self.alphabeta(depth+1,currBoardIdx,alpha,beta,False)

                currBoardIdx = preBoardIdx
                if value>bestValue:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                alpha = max(alpha,bestValue)
                self.board[i[0]][i[1]] = "_"
                if(beta<=alpha):
                    break

        elif(isMax == False):
            if (self.checkWinner()==-1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax) ###check later

            elif (self.checkWinner()==1):
                if (depth==1 or depth==3):
                    return self.evaluatePredifined(not isMax)
                else: 
                    return self.evaluatePredifined(isMax)

            bestValue = 100000000
            moveSequence = self.checkMoveList(currBoardIdx)
            for i in moveSequence:
                self.board[i[0]][i[1]] = "O"
                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == self.maxDepth:
                    currBoardIdx = preBoardIdx
                value = self.alphabeta(depth+1,currBoardIdx,alpha,beta,True)
                currBoardIdx = preBoardIdx
                if bestValue>value:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                beta = min(bestValue,beta)
                self.board[i[0]][i[1]] = "_"
                if(beta<=alpha):
                    break
        return bestValue

    def alphabeta2(self,depth,currBoardIdx,alpha,beta,isMax): # alphabeta2 is used for my designed agent
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        bestValue=0.0
        preBoardIdx = 0
        if (self.checkWinner()==1 or self.checkWinner()==-1):
            return self.evaluateDesigned(isMax)
        if (depth==4):
            self.count += 1
            return self.evaluateDesigned(isMax)
        if(isMax==True):
            bestValue = -100000000
            moveList = self.checkMoveList(currBoardIdx)
            for i in moveList:
                self.board[i[0]][i[1]] = "X"

                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == 4:
                    currBoardIdx = preBoardIdx

                value = self.alphabeta2(depth+1,currBoardIdx,alpha,beta,False)

                currBoardIdx = preBoardIdx
                if value>bestValue:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                alpha = max(alpha,bestValue)
                self.board[i[0]][i[1]] = "_"
                if(beta<=alpha):
                    break
        elif(isMax == False):
            bestValue = 100000000
            moveSequence = self.checkMoveList(currBoardIdx)
            for i in moveSequence:
                self.board[i[0]][i[1]] = "O"
                local_corner = self.globalIdx[currBoardIdx]
                preBoardIdx = currBoardIdx

                currBoardIdx = (i[0]-local_corner[0])*3 + i[1]-local_corner[1]
                if (depth+1) == 4:
                    currBoardIdx = preBoardIdx
                value = self.alphabeta2(depth+1,currBoardIdx,alpha,beta,True)
                currBoardIdx = preBoardIdx
                if bestValue>value:
                    bestValue = value
                    if depth == 0:
                        self.point = i
                beta = min(bestValue,beta)
                self.board[i[0]][i[1]] = "_"
                if(beta<=alpha):
                    break

        return bestValue

    
    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive): 
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        winner=0
        expandedNodes=[]
        currBoardIdx=self.startBoardIdx
        alpha=-inf
        beta=inf
        self.count=0

        while self.checkMovesLeft(currBoardIdx):
            self.count=0
            if maxFirst:
                bestvalue=self.minimax(0, currBoardIdx,maxFirst) if isMinimaxOffensive else self.alphabeta(0, currBoardIdx,alpha,beta,maxFirst)
                self.board[self.point[0]][self.point[1]]="X"
            
            else:
                bestvalue=self.minimax(0, currBoardIdx,maxFirst) if isMinimaxDefensive else self.alphabeta(0, currBoardIdx,alpha,beta,maxFirst)               
                self.board[self.point[0]][self.point[1]]="O"


            maxFirst= not(maxFirst)

            local_corner = self.globalIdx[currBoardIdx]
            currBoardIdx = (self.point[0]-local_corner[0])*3 + self.point[1]-local_corner[1]
      
            bestMove.append(self.point)
            bestValue.append(bestvalue)
            gameBoards.append(self.board)
            expandedNodes.append(self.count)
            
            winner=self.checkWinner()
            if winner!=0:
                break
        return gameBoards, bestMove, expandedNodes, bestValue, winner



    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the bwinner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE

        bestMove=[]
        gameBoards=[]
        winner=0
        currBoardIdx=randint(0,8)
        print("currBoardIdx is", currBoardIdx)
        isMax=randint(0,1)
        print("isMax is", isMax)
        alpha=-inf
        beta=inf
        expandedNodes=[]
        self.count=0

        while self.checkMovesLeft(currBoardIdx):
            if isMax:
                self.count=0
                bestvalue=self.alphabeta(0, currBoardIdx,alpha,beta,isMax) #alphabeta is predefinied agent
                self.board[self.point[0]][self.point[1]]="X"
                expandedNodes.append(self.count)

            else: 
                self.count=0
                bestvalue=self.alphabeta2(0, currBoardIdx,alpha,beta,isMax) #alphabeta2 is my agent
                self.board[self.point[0]][self.point[1]]="O"
                expandedNodes.append(self.count)

            isMax=1-isMax
            local_corner = self.globalIdx[currBoardIdx]
            currBoardIdx = (self.point[0]-local_corner[0])*3 + self.point[1]-local_corner[1]
            bestMove.append(self.point)
            gameBoards.append(self.board)            
            winner=self.checkWinner()
            if winner!=0:
                break
        print("expandedNodes is", expandedNodes)
        return gameBoards, bestMove, winner

    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        currBoardIdx=randint(0,8)
        isMax=randint(0,1)
        alpha=-inf
        beta=inf
        validInput=0
        print("The initial board is:")
        self.printGameBoard() 
        while self.checkMovesLeft(currBoardIdx):
            if isMax:
                while not validInput:
                    print("Current board index is",currBoardIdx)
                    print("The possible move is between", self.globalIdx[currBoardIdx][0], "," ,self.globalIdx[currBoardIdx][1], " and ", self.globalIdx[currBoardIdx][0]+2,",",self.globalIdx[currBoardIdx][1]+2)
                    coordinate=tuple(input('Please input non-occupied coordinate in forms of "x,y":'))

                    if (self.board[int(coordinate[0])][int(coordinate[2])]=="_" and int(coordinate[0])-self.globalIdx[currBoardIdx][0]>=0 and int(coordinate[0])-self.globalIdx[currBoardIdx][0]<=2 and int(coordinate[2])-self.globalIdx[currBoardIdx][1]>=0 and int(coordinate[2])-self.globalIdx[currBoardIdx][1]<=2): #
                        self.point=(int(coordinate[0]),int(coordinate[2]))
                        print(self.point)
                        validInput=1
                    else: 
                        print("Invalid input, please re-enter")
                self.board[self.point[0]][self.point[1]]="X"
                validInput=0

            else: 
                bestvalue=self.alphabeta2(0, currBoardIdx,alpha,beta,isMax)
                self.board[self.point[0]][self.point[1]]="O"
                print("Computer moves at",self.point)
                print("After computer moving,")
            
            isMax=1-isMax
            local_corner = self.globalIdx[currBoardIdx]
            currBoardIdx = (self.point[0]-local_corner[0])*3 + self.point[1]-local_corner[1]

            bestMove.append(self.point)
            gameBoards.append(self.board)
            print("Current board is:")
            self.printGameBoard()
            winner=self.checkWinner()
            if winner!=0:
                break

        return gameBoards, bestMove, winner

    

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,False,False) #return gameBoards, bestMove, expandedNodes, bestValue, winner
    #gameBoards, bestMove, winner=uttt.playGameYourAgent() #return gameBoards, bestMove, winner
    #gameBoards, bestMove, winner=uttt.playGameHuman() #return gameBoards, bestMove, winner
    uttt.printGameBoard()
    print("bestMove is", bestMove)
    #print("expandedNodes is", expandedNodes) #Only for uttt.playGamePredifinedAgent
    #print("bestValue is", bestValue) #Only for uttt.playGamePredifinedAgent

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")