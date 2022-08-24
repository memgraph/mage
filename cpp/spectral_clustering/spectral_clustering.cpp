// Spectural Clustering.cpp : Defines the entry point for the application.

#include "spectral_clustering.h"
#include <vector>


using namespace std;

// Steps 1: Lalpacian Matrix Formula : L = D−W

void createAdjacencyMatrix(vector<vector<int>>& adjacencyMatrix, int verticies) {
    int temp;
    vector<int> row;
    for (int k = 1; k < verticies + 1; k++) {
        cout << "Enter all elements from row " << k << ", seperated by a space: ";
        for (int i = 0; i < verticies; i++) {
            cin >> temp;
            row.push_back(temp);
        }
        adjacencyMatrix.push_back(row);
        row.clear();
    }
}

void createDegreeMatrix(vector<vector<int>>& degreeMatrix, vector<vector<int>> adjacencyMatrix) {
    for (int i = 0; i < adjacencyMatrix.size(); i++)
    {
        int sum = 0;
        vector<int> row(adjacencyMatrix.size(), 0);

        for (int j = 0; j < adjacencyMatrix[i].size(); j++)
        {
            sum += adjacencyMatrix[i][j];
        }

        row[i] = sum;
        degreeMatrix.push_back(row);
    }
}

void createLaplacianMatrix(vector<vector<int>>& laplacianMatrix, vector<vector<int>> degreeMatrix, vector<vector<int>> adjacencyMatrix) {
    int difference;
    vector<int> row;

    for (int i = 0; i < degreeMatrix.size(); i++)
    {
        for (int j = 0; j < degreeMatrix[i].size(); j++)
        {
            difference = degreeMatrix[i][j] - adjacencyMatrix[i][j];
            row.push_back(difference);
        }
        laplacianMatrix.push_back(row);
        row.clear();
    }
}

void printMatrix(vector<vector<int>> matrix) {
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[i].size(); j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main()
{
    int verticies = 3;
    vector<vector<int>> adjacencyMatrix;
    vector<vector<int>> degreeMatrix;
    vector<vector<int>> laplacianMatrix;

    createAdjacencyMatrix(adjacencyMatrix, verticies);
    cout << "\nAdjacency Matrix:\n";
    printMatrix(adjacencyMatrix);

    createDegreeMatrix(degreeMatrix, adjacencyMatrix);
    createLaplacianMatrix(laplacianMatrix, degreeMatrix, adjacencyMatrix);

    cout << "Degree Matrix:\n";
    printMatrix(degreeMatrix);
    cout << "Laplacian Matrix:\n";
    printMatrix(laplacianMatrix);

	
	return 0;
}
