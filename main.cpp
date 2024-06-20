#include <iostream>
#include <vector>

const int SIZE = 8;

// 检查是否在同一对角线上
bool isDiagonal(int row1, int col1, int row2, int col2) {
    return std::abs(row1 - row2) == std::abs(col1 - col2);
}

// 检查放置皇后是否安全
bool isSafe(std::vector<int>& board, int row, int col) {
    for (int i = 0; i < row; ++i) {
        // 检查同一列和对角线
        if (board[i] == col || isDiagonal(i, board[i], row, col)) {
            return false;
        }
    }
    return true;
}

// 打印棋盘
void printSolution(std::vector<int>& board) {
    static int count = 0;
    std::cout << "Solution " << ++count << ":" << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            std::cout << (board[i] == j ? "Q " : "._");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 递归函数解决N皇后问题
bool solveNQueens(std::vector<int>& board, int row) {
    if (row == SIZE) {
        // 找到一个解决方案
        printSolution(board);
        return true;
    }

    bool flag = false;
    for (int i = 0; i < SIZE; ++i) {
        // 尝试在当前行的每一列放置皇后
        if (isSafe(board, row, i)) {
            board[row] = i;
            if (solveNQueens(board, row + 1)) {
                flag = true;
            }
            // 回溯
            board[row] = -1; // 清除列
        }
    }

    return flag;
}

int main() {
    std::vector<int> board(SIZE, -1); // 初始化棋盘，-1 表示皇后尚未放置
    if (solveNQueens(board, 0)) {
        std::cout << "Solution found!" << std::endl;
    } else {
        std::cout << "No solution exists." << std::endl;
    }
    return 0;
}