from PIL import Image, ImageChops, ImageDraw
import numpy as np


class Agent:

    def __init__(self):
        pass

    def Solve(self, problem):
        if problem.problemType == '2x2':
            return self.solve_problem_2x2(problem)
        elif problem.problemType == '3x3':
            return self.solve_problem_3x3(problem)
        else:
            return -1

    def solve_problem_3x3(self, problem):
        # open images A-H,1-8 and convert them all to back/white/grey mode
        A = Image.open(problem.figures['A'].visualFilename).convert('L')
        B = Image.open(problem.figures['B'].visualFilename).convert('L')
        C = Image.open(problem.figures['C'].visualFilename).convert('L')
        D = Image.open(problem.figures['D'].visualFilename).convert('L')
        E = Image.open(problem.figures['E'].visualFilename).convert('L')
        F = Image.open(problem.figures['F'].visualFilename).convert('L')
        G = Image.open(problem.figures['G'].visualFilename).convert('L')
        H = Image.open(problem.figures['H'].visualFilename).convert('L')
        opList = []
        for i in range(1, 9):
            fig = Image.open(problem.figures.get(str(i)).visualFilename).convert('L')
            opList.append(fig)

        finalScore = self.final_score_3x3(A, B, C, D, E, F, G, H, opList)
        maxScore = max(finalScore)
        answer = finalScore.index(maxScore) + 1
        return answer

    def final_score_3x3(self, A, B, C, D, E, F, G, H, opList):  # [x, x, x, x, x, x, x, x]
        match = np.array(self.match_score(A, opList))
        union = np.array(self.union_pixel_density(A, B, C, D, E, F, G, H, opList))
        reflection = np.array(self.reflection_score_3x3(A, B, C, G, opList))
        rotation = np.array(self.rotation_score_3x3(A, C, G, opList))
        symmetry = np.array(self.split_score(A, B, C, G, H, opList))
        difference = np.array(self.compare_difference_general(C, F, opList)) + \
                     np.array(self.compare_difference_general(G, H, opList)) + \
                     np.array(self.compare_difference_setE(A, B, C, G, H, opList))

        scoreList = list(match + union + reflection + rotation + symmetry + 2 * difference)

        if self.same_image(A, B, 0.01) and self.same_image(B, C, 0.01): # for solving an exception of C-01: A=B=C
            scoreList = list(match + union + reflection + rotation + symmetry)

        #print(scoreList)
        return scoreList

    def black_pixel_density(self, image):
        black = 0
        pixels = image.getdata()
        for pixel in pixels:
            if pixel == 0:  # black
                black += 1
        density = black / len(list(pixels))
        return density

    def match_score(self, figure, opList):  # [x, x, x, x, x, x, x, x]
        matchScore = []
        for op in opList:
            if self.same_image(figure, op, 1):
                matchScore.append(1)
            else:
                matchScore.append(0)
        return matchScore

    def union_pixel_density(self, A, B, C, D, E, F, G, H, opList):  # [x, x, x, x, x, x, x, x]
        row1 = self.black_pixel_density(A) + self.black_pixel_density(B) + self.black_pixel_density(C)
        row2 = self.black_pixel_density(D) + self.black_pixel_density(E) + self.black_pixel_density(F)
        row3 = self.black_pixel_density(G) + self.black_pixel_density(H)
        unionScore = []

        if abs(row1 - row2) < 0.002:  # black pixel density: row1 = row2
            for op in opList:
                if abs(row3 + self.black_pixel_density(op) - row1) < 0.002 and \
                        abs(row3 + self.black_pixel_density(op) - row2) < 0.002:  # black pixel density: row1=row2=row3
                    unionScore.append(10)
                else:
                    unionScore.append(0)
        elif abs(row1 - row2) < 0.012:
            for op in opList:
                if abs(row3 + self.black_pixel_density(op) - row1) < 0.037 and \
                        abs(row3 + self.black_pixel_density(op) - row2) < 0.037:
                    unionScore.append(8)
                elif abs(row3 + self.black_pixel_density(op) - row1) < 0.05 and \
                        abs(row3 + self.black_pixel_density(op) - row2) < 0.05:
                    unionScore.append(4)
                else:
                    unionScore.append(0)
        else:
            unionScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return unionScore

    def reflection_score_3x3(self, A, B, C, G, opList):  # [x, x, x, x, x, x, x, x]
        lrScore = []  # left-right
        lrFigA = A.transpose(Image.FLIP_LEFT_RIGHT)
        lrFigG = G.transpose(Image.FLIP_LEFT_RIGHT)
        pr_A = self.pixel_ratio(A)
        pr_C = self.pixel_ratio(C)
        pr_G = self.pixel_ratio(G)

        # overlap problems only, like E-10,11, exclude unchanged images
        if self.same_image(lrFigA, B, 0.03) and (pr_C - pr_A) > 0.12:
            for op in opList:
                if (self.pixel_ratio(op) - pr_G) > 0.12:
                    lrScore.append(15)
                else:
                    lrScore.append(0)
        # reflection problems, including unchanged images
        elif self.same_image(lrFigA, C, 0.05):
            for op in opList:
                if self.same_image(lrFigG, op, 0.05):
                    lrScore.append(1)
                else:
                    lrScore.append(0)
        else:
            lrScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return lrScore

    def rotation_score_3x3(self, A, C, G, opList):  # [x, x, x, x, x, x, x, x]
        rotScore = []
        rotA = A.rotate(270) # clockwise 90 degree
        lrA = A.transpose(Image.FLIP_LEFT_RIGHT)

        if self.same_image(lrA, C, 0.027): # exclude reflections
            rotScore = [0, 0, 0, 0, 0, 0, 0, 0]
        elif self.same_image(rotA, C, 0.027):
            rotG = G.rotate(270)
            for op in opList:
                if self.same_image(rotG, op, 0.027):
                    rotScore.append(8)
                else:
                    rotScore.append(0)
        else:
            rotScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return rotScore

    def split_lr(self, image1, image2):  # split image by left and right
        # crop the image: the first two coordinates in "box" (x, x, x, x) is start position (upper left corner),
        # the last two coordinates in "box" is the end position (bottom right corner)
        width1, height1 = image1.size
        box1 = (0, 0, width1 / 2, height1)
        box2 = (width1 / 2, 0, width1, height1)
        left_1 = image1.crop(box1)
        right_1 = image1.crop(box2)

        width2, height2 = image2.size
        box3 = (0, 0, width2 / 2, height2)
        box4 = (width2 / 2, 0, width2, height2)
        left_2 = image2.crop(box3)
        right_2 = image2.crop(box4)

        if self.same_image(left_1, right_2, 0.12) and self.same_image(right_1, left_2, 0.12):
            return True
        else:
            return False

    def split_td(self, image1, image2):  # split image by top and down
        # crop the image: the first two coordinates in "box" (x, x, x, x) is start position (upper left corner),
        # the last two coordinates in "box" is the end position (bottom right corner)
        width1, height1 = image1.size
        box1 = (0, 0, width1, height1 / 2)
        box2 = (0, height1 / 2, width1, height1)
        top_1 = image1.crop(box1)
        down_1 = image1.crop(box2)

        width2, height2 = image2.size
        box3 = (0, 0, width2, height2 / 2)
        box4 = (0, height2 / 2, width2, height2)
        top_2 = image2.crop(box3)
        down_2 = image2.crop(box4)

        if self.same_image(top_1, top_2, 0.03):
            return "top"
        elif self.same_image(down_1, down_2, 0.03):
            return "down"
        else:
            return False

    def split_score(self, A, B, C, G, H, opList):  # [x, x, x, x, x, x, x, x]
        lrScore = []
        tdScore = []

        if self.split_lr(A, C):  # symmetry in left and right
            tdScore = [0, 0, 0, 0, 0, 0, 0, 0]
            for op in opList:
                if self.split_lr(G, op):
                    lrScore.append(5)
                else:
                    lrScore.append(0)
        else:  # symmetry in top and down
            lrScore = [0, 0, 0, 0, 0, 0, 0, 0]
            if self.split_td(A, C) == "top" and self.split_td(B, C) == "down":
                for op in opList:
                    if self.split_td(G, op) == "top" and self.split_td(H, op) == "down":
                        tdScore.append(5)
                    else:
                        tdScore.append(0)
            elif self.split_td(A, C) == "down" and self.split_td(B, C) == "top":
                for op in opList:
                    if self.split_td(G, op) == "down" and self.split_td(H, op) == "top":
                        tdScore.append(5)
                    else:
                        tdScore.append(0)
            else:
                lrScore = [0, 0, 0, 0, 0, 0, 0, 0]
                tdScore = [0, 0, 0, 0, 0, 0, 0, 0]

        splitScore = list(np.array(lrScore) + np.array(tdScore))
        return splitScore

    # difference comparing in image C,F for solving general 3x3 Problem C&D&E
    def compare_difference_general(self, C, F, opList):  # [x, x, x, x, x, x, x, x]
        pr_C = self.pixel_ratio(C)
        pr_F = self.pixel_ratio(F)
        diff = ImageChops.difference(C, F)
        diffScore = []

        if (pr_C - pr_F) > 0.23:  # white/black pixel_ratio decrease (black pixel increase) with row/column
            for op in opList:
                diff1 = ImageChops.difference(F, op)
                if (pr_F - self.pixel_ratio(op)) > 0.4:  # optimal threshold for Problems D and E
                    # if (pr_F - self.pixel_ratio(op)) > 0.36:  # optimal threshold for Problems C
                    if self.same_image(diff1, diff, 0.037):
                        diffScore.append(15)
                    elif self.same_image(diff1, diff, 0.06):
                        diffScore.append(10)
                    elif self.same_image(diff1, diff, 0.08):
                        diffScore.append(5)
                    elif self.same_image(diff1, diff, 0.1):
                        diffScore.append(3)
                    elif self.same_image(diff1, diff, 0.2):
                        diffScore.append(2)
                    else:
                        diffScore.append(1)
                else:
                    diffScore.append(0)

        elif (pr_F - pr_C) > 0.27:  # white/black pixel_ratio increase (black pixel decrease) with row/column
            for op in opList:
                diff1 = ImageChops.difference(F, op)
                if (self.pixel_ratio(op) - pr_F) > 0.4:  # optimal threshold for Problems D and E
                    # if (self.pixel_ratio(op) - pr_F) > 0.36: # optimal threshold for Problems C
                    if self.same_image(diff1, diff, 0.037):
                        diffScore.append(15)
                    elif self.same_image(diff1, diff, 0.06):
                        diffScore.append(10)
                    elif self.same_image(diff1, diff, 0.08):
                        diffScore.append(5)
                    elif self.same_image(diff1, diff, 0.1):
                        diffScore.append(3)
                    elif self.same_image(diff1, diff, 0.2):
                        diffScore.append(2)
                    else:
                        diffScore.append(1)
                else:
                    diffScore.append(0)
        else:
            diffScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return diffScore

    # difference comparing in image A,B,C for solving special 3x3 Problem E
    def compare_difference_setE(self, A, B, C, G, H, opList):  # [x, x, x, x, x, x, x, x]
        diffScore = []
        diff1 = ImageChops.difference(A, B)
        # if image A = B, diff1 should be completely black or so, because the rgb value of black is {0,0,0}.
        # diff1.show()

        if self.same_image(ImageChops.invert(diff1), C, 0.04):  # solving problems have image A-B=C or B-A=C or A+B=C
            for op in opList:
                diff2 = ImageChops.difference(G, H)
                # if self.same_image(ImageChops.invert(diff2), op, 0.04): # optimal threshold for Problems C
                if self.same_image(ImageChops.invert(diff2), op, 0.06):  # optimal threshold for Problems D and E
                    diffScore.append(15)
                else:
                    diffScore.append(0)
        else:
            diffScore = [0, 0, 0, 0, 0, 0, 0, 0]
        return diffScore

    def solve_problem_2x2(self, problem):
        # open images A,B,C,1-6 and convert them all to back/white/grey mode
        figA = Image.open(problem.figures['A'].visualFilename).convert('L')
        figB = Image.open(problem.figures['B'].visualFilename).convert('L')
        figC = Image.open(problem.figures['C'].visualFilename).convert('L')
        opList = []
        for i in range(1, 7):
            fig = Image.open(problem.figures.get(str(i)).visualFilename).convert('L')
            opList.append(fig)

        finalScore = self.final_score_2x2(figA, figB, figC, opList)
        maxScore = max(finalScore)
        answer = finalScore.index(maxScore) + 1
        return answer

    def final_score_2x2(self, figA, figB, figC, opList):  # [x, x, x, x, x, x] in which x = 0 or 1
        refAB = self.reflection_score(figA, figB, figC, opList)
        refAC = self.reflection_score(figA, figC, figB, opList)

        rotAB = self.rotation_score(figA, figB, figC, opList)
        rotAC = self.rotation_score(figA, figC, figB, opList)

        diffAB = self.compare_difference(figA, figB, figC, opList)
        diffAC = self.compare_difference(figA, figC, figB, opList)

        fillAB = self.fill_score(figA, figB, figC, opList)
        fillAC = self.fill_score(figA, figC, figB, opList)

        reflection = np.array(refAB) + np.array(refAC)
        rotation = np.array(rotAB) + np.array(rotAC)
        difference = np.array(diffAB) + np.array(diffAC)
        imageFill = np.array(fillAB) + np.array(fillAC)

        scoreList = list(2 * reflection + rotation + difference + imageFill)

        #print(scoreList)
        return scoreList

    def pixel_ratio(self, image):  # white/black pixel ratio
        black = 1
        white = 1
        pixels = image.getdata()
        for pixel in pixels:
            if pixel == 0:  # black
                black += 1
            else:
                white += 1
        return white / black

    def same_image(self, image1, image2, threshold):
        diff = ImageChops.difference(image1, image2)
        if self.pixel_ratio(diff) < threshold:
            return True
        else:
            return False

    def reflection_score(self, figA, figB, figC, opList):  # [x, x, x, x, x, x] in which x = 0 or 1
        lrScore = []  # left-right
        tdScore = []  # top-down
        lrFigA = figA.transpose(Image.FLIP_LEFT_RIGHT)
        tdFigA = figA.transpose(Image.FLIP_TOP_BOTTOM)

        if self.same_image(lrFigA, figB, 0.05):
            tdScore = [0, 0, 0, 0, 0, 0]
            lrFigC = figC.transpose(Image.FLIP_LEFT_RIGHT)
            for op in opList:
                if self.same_image(lrFigC, op, 0.05):
                    lrScore.append(1)
                else:
                    lrScore.append(0)
        elif self.same_image(tdFigA, figB, 0.18):
            lrScore = [0, 0, 0, 0, 0, 0]
            tdFigC = figC.transpose(Image.FLIP_TOP_BOTTOM)
            for op in opList:
                if self.same_image(tdFigC, op, 0.01):
                    tdScore.append(2)
                elif self.same_image(tdFigC, op, 0.18):
                    tdScore.append(1)
                else:
                    tdScore.append(0)
        else:
            lrScore = [0, 0, 0, 0, 0, 0]
            tdScore = [0, 0, 0, 0, 0, 0]

        refScore = list(np.array(lrScore) + np.array(tdScore))
        return refScore

    def rotation_score(self, figA, figB, figC, opList):  # [x, x, x, x, x, x] in which x = 0 or 1
        rotScore = []
        rotFigA = figA.rotate(270) # clockwise 90 degrees

        if self.same_image(rotFigA, figB, 0.1):
            rotFigC = figC.rotate(270)
            for op in opList:
                if self.same_image(rotFigC, op, 0.06):
                    rotScore.append(3)
                elif self.same_image(rotFigC, op, 0.1):
                    rotScore.append(1)
                else:
                    rotScore.append(0)
        else:
            rotScore = [0, 0, 0, 0, 0, 0]
        return rotScore

    def compare_difference(self, figA, figB, figC, opList): # [x, x, x, x, x, x] in which x = 0 or 1
        diffScore = []
        diffAB = ImageChops.difference(figA, figB)

        for op in opList:
            diffCOp = ImageChops.difference(figC, op)
            if self.same_image(diffAB, diffCOp, 0.037):
                diffScore.append(1)
            else:
                diffScore.append(0)
        return diffScore

    def image_fill(self, image):
        width, height = image.size
        center = (int(0.5 * width), int(0.5 * height))
        ImageDraw.floodfill(image, xy = center, value = 0) # value = black = 0
        return image

    # for solving Basic Problem B-09
    def fill_score(self, FigA, FigB, FigC, opList): # [x, x, x, x, x, x]
        fillScore = []
        copy_A = FigA.copy()
        copy_C = FigC.copy()
        copy_opList = []
        for op in opList:
            copy_opList.append(op.copy())

        if self.pixel_ratio(FigA) > self.pixel_ratio(FigB):
            fill_A = self.image_fill(copy_A)
            if self.same_image(fill_A, FigB, 0.05):
                fill_C = self.image_fill(copy_C)
                for op in copy_opList:
                    if self.same_image(fill_C, op, 0.05):
                        fillScore.append(5)
                    else:
                        fillScore.append(0)
            else:
                fillScore = [0, 0, 0, 0, 0, 0]
        else:
            fillScore = [0, 0, 0, 0, 0, 0]

        return fillScore
