#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

//Function to input y values from CSV file into a vector
void yValueInput(const string &fileName, vector<vector<double>> &yValuesVec, vector<string> &symbolVec)
{
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << fileName << endl;
        return;
    }

    string line; 

    while(getline(file, line))
    {
        //Initialization
        stringstream ss(line);
        string value;
        vector<double> yValues;
        string symbol;

        bool invalidLine = false;//Flag to check invalid line

        //Read first field as symbol
        getline(ss, symbol, ',');

        //Reading remaining fields as yValues
        while(getline(ss, value, ','))
        {
            try
            {
                double input = stod(value);
                yValues.push_back(input);
            }
            catch(const invalid_argument &e)
            {
                cerr << "warning: Invalid argument in line: " << line << endl;
                cerr << "Value: " << value << endl;
                invalidLine = true;
                break;
            }
        } 

        //Append to vectors if valid
        if (!invalidLine)
        {
            symbolVec.push_back(symbol);
            yValuesVec.push_back(yValues);
        }


    }

    file.close();
}

//Function to transpose a matrix
vector<vector<double>> transpose(const vector<vector<double>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<double>> tX(cols, vector<double>(rows));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            tX[j][i] = matrix[i][j];
        }
    }

    return tX;
}

//Function to multiply two matrices
vector<vector<double>> multiplyMatrices(const vector<vector<double>> &X, const vector<vector<double>> &Y) 
{
    int rowsX = X.size();
    int colsX = X[0].size();
    int colsY = Y[0].size();

    vector<vector<double>> result(rowsX, vector<double>(colsY, 0));

    for (int i = 0; i < rowsX; i++) 
    {
        for (int j = 0; j < colsY; j++) 
        {
            for (int k = 0; k < colsX; k++) 
            {
                result[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
    return result;
}

//Function to invert 2x2 matrix
vector<vector<double>> invert2x2(const vector<vector<double>>& a) 
{
    double determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    vector<vector<double>> inverse(2, vector<double>(2));

    inverse[0][0] = a[1][1] / determinant;
    inverse[0][1] = -a[0][1] / determinant;
    inverse[1][0] = -a[1][0] / determinant;
    inverse[1][1] = a[0][0] / determinant;

    return inverse;
}

//Function to multiply a matrix and a vector
vector<double> multiplyMatrixVector(const vector<vector<double>> &matrix, const vector<double> &vec) 
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<double> result(rows, 0);

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

//Function to find weights for model
vector<double> weightVal(vector<vector<double>> xValues, vector<double> yVec)
{
    vector<vector<double>> Xt = transpose(xValues);
    vector<vector<double>> XtX = multiplyMatrices(Xt, xValues);
    vector<vector<double>> invXtX = invert2x2(XtX);
    vector<vector<double>> invXtXXt = multiplyMatrices(invXtX, Xt);

    // Calculate weights w using invXtXXt and Y_matrix
    vector<double> w_matrix = multiplyMatrixVector(invXtXXt, yVec);

    return w_matrix;
}

//Function to fit a linear model and make a prediction
void linearPredict(vector<vector<double>> xValues, vector<double> yVec)
{
    vector<double> w = weightVal(xValues, yVec);
    //Create the next interval
    double nextX = xValues.size() + 1;
    
    // Predict for the next quarter using the weights
    double prediction = w[0] + nextX * w[1];

    cout << "w = " << w[0] << " , " << w[1] << endl;
    cout << "Linear prediction for next quarter = " << prediction << endl;
}

//Function to fit a logarithmic model and make a prediction
void logPredict(vector<vector<double>> xValues, vector<double> yVec)
{
    vector<vector<double>> logX(xValues.size(), vector<double>(2));
    for (int i = 0; i < xValues.size(); i++)
    {
        logX[i][0] = 1;
        logX[i][1] = log(xValues[i][1]);
    }

    vector<double> w = weightVal(logX, yVec);

    //Create the next interval
    double nextX = xValues.size() + 1;

    // Predict for the next quarter using the weights
    double prediction = w[0] + log(nextX) * w[1];

    cout << "w = " << w[0] << " , " << w[1] << endl;
    cout << "Logrithmic prediction for next quarter = " << prediction << endl;
}

//Function to fit an exponential model and make a prediction
void expPredict(vector<vector<double>> xValues, vector<double> yVec)
{
    for (int i = 0; i < yVec.size(); i++)
    {
        yVec[i] = log(yVec[i]);
    }

    vector<double> w = weightVal(xValues, yVec);

    //Create the next interval
    double nextX = xValues.size() + 1;

    // Predict for the next quarter using the weights
    double prediction = exp(w[0]) * exp(nextX * w[1]);

    cout << "w = " << w[0] << " , " << w[1] << endl;
    cout << "Exponential prediction for next quarter = " << prediction << endl;
}

//Function to fit a power model and make a prediction
void powPredict(vector<vector<double>> xValues, vector<double> yVec)
{
    vector<vector<double>> logX(xValues.size(), vector<double>(2));
    for (int i = 0; i < xValues.size(); i++)
    {
        logX[i][0] = 1;
        logX[i][1] = log(xValues[i][1]);
    }

    for (int i = 0; i < yVec.size(); i++)
    {
        yVec[i] = log(yVec[i]);
    }

    vector<double> w = weightVal(logX, yVec);

    //Create the next interval
    double nextX = xValues.size() + 1;

    // Predict for the next quarter using the weights
    double prediction = exp(w[0]) * pow(nextX, w[1]);

    cout << "w = " << w[0] << " , " << w[1] << endl;
    cout << "Power Curve prediction for next quarter = " << prediction << endl;
}


int main()
{
    //Initialize vector to hold the y values and symbol
    vector<vector<double>> yValues;
    vector<string> symbol;

    //Create an output stream to a log file in append mde
    ofstream logFile("log.txt", ios::app);

    //Redirect cout to log
    streambuf* originalCoutBuffer = cout.rdbuf();
    cout.rdbuf(logFile.rdbuf());

    for (int i = 0; i < 3; i++)
    {
        //Load data from csv file as data
        switch(i)
        {
            case 0:
                yValueInput("dividends.csv", yValues,symbol);
                cout << "Dividends" << endl;
                cout << "======================================" << endl;
                break;
            case 1:
                yValueInput("earnings.csv", yValues,symbol);
                cout << "Earnings" << endl;
                cout << "======================================" << endl;
                break;
            case 2:
                yValueInput("revenues.csv", yValues,symbol);
                cout << "Revenues" << endl;
                cout << "======================================" << endl;
                break;
            default:
                cout << "No input" << endl;
                break;
        }

        //Prepare quarters
        vector<vector<double>> xValues;
        for (int i = 0; i < yValues[0].size(); i++) 
        {
            vector<double> rowV;
            rowV.push_back(1);
            rowV.push_back(i + 1);

            xValues.push_back(rowV);
        }

        for (int i = 0; i < yValues.size(); i++)
        {
            cout << "Company: " << symbol[i] << endl;
            linearPredict(xValues,yValues[i]);
            logPredict(xValues,yValues[i]);
            expPredict(xValues,yValues[i]);
            powPredict(xValues,yValues[i]);
            cout << endl;
        }
        
        //Clears vectors
        yValues.clear();
        symbol.clear();
    }

    //Restore cout and close log file
    cout.rdbuf(originalCoutBuffer);
    logFile.close();

    return 0;
}