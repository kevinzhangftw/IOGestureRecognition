//
//  main.cpp
//  ImportStuff
//
//  Created by Kevin Zhang on 2016-10-28.
//  Copyright © 2016 Kevin Zhang. All rights reserved.
//

#include <iostream>
#include <GRT/GRT.h>
#include "opencv2/core/version.hpp"

using namespace std;
using namespace GRT;

void runMatrixExample();
void runClassificationDataExample();

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "GRT Version: " << GRT_VERSION << endl;
    cout << "OpenCV Version: " << CV_VERSION << endl;
    
    runMatrixExample();
    runClassificationDataExample();
    
    return EXIT_SUCCESS;
}

void runMatrixExample(){
    //This example demonstrates how to save and load data from a matrix
    //Create dummy matrix and fill it with some data
    //The matrix will have 100 rows and 5 columns
    MatrixFloat x( 100, 5 );
    Random random;
    for(unsigned int i=0; i<x.getNumRows(); i++){
        for(unsigned int j=0; j<x.getNumCols(); j++){
            x[i][j] = random.getRandomNumberUniform( 0.0, 1.0 );
        }
    }
    cout << "saving data to CSV file...\n";
    //Save the dummy data to a CSV file
    //Each row in the CSV file corresponds to a row in the matrix
    //Columns will be seperated by the comma delimiter (',')
    x.save( "matrix_data.csv" );
    //Create a second matrix to load the data we just saved
    MatrixFloat y;
    cout << "loading data from CSV file...\n";
    //Load the data from the CSV file, the load function will automatically determine the size of the matrix
    y.load( "matrix_data.csv" );
    cout << "validating data...\n";
    //Check to make sure the sizes match
    grt_assert( x.getSize() == y.getSize() );
    grt_assert( x.getNumRows() == y.getNumRows() );
    grt_assert( x.getNumCols() == y.getNumCols() );
    //Check the values match
    for(unsigned int i=0; i<x.getNumRows(); i++){
        for(unsigned int j=0; j<x.getNumCols(); j++){
            grt_assert( fabs( x[i][j] - y[i][j] ) < 1.0e-5 );
        }
    }
    cout << "data OK\n";
}

void runClassificationDataExample(){
    //This example demonstrates how to save and load supervised classification data
    //Create a dummy classification data structure
    ClassificationData data;
    const unsigned int numDimensions = 3;
    const unsigned int numExamples = 1000;
    VectorFloat sample( numDimensions );
    Random random;
    //Set the size of the classification data structure, this will make adding new samples faster
    data.setNumDimensions( numDimensions );
    data.reserve( numExamples );
    //Add some random data to the classification data structure
    for(unsigned int i=0; i<numExamples; i++){
        for(unsigned int j=0; j<numDimensions; j++){
            sample[j] = random.getRandomNumberUniform( 0.0, 1.0 );
        }
        //Create a random class label, between 0 and 10
        unsigned int classLabel = random.getRandomNumberInt( 0, 10 );
        //Add the sample to the data structure
        data.addSample( classLabel, sample );
    }
    //Save the classification data to a custom GRT format (this is faster to save/load)
    //The is a basic ASCII file, with a custom header followed by the raw data
    data.save( "classification_data.grt" );
    //The data can also be saved to a CSV file
    //Each row in the CSV file corresponds to a sample in the classification data
    //The first column in the CSV file corresponds to the class label
    //The following columns correspond to each of the dimensions in the input data
    //There will therefore be M rows and N+1 columns, where M is the number of examples and N is the number of dimensions
    //Columns will be seperated by the comma delimiter (',')
    data.save( "classification_data.csv" );
    
    //Create a second classification data structure to load the GRT formatted data
    ClassificationData grtData;
    //Load the data from the GRT file
    grtData.load( "classification_data.grt" );
    //Check to make sure the sizes match
    grt_assert( data.getNumDimensions() == grtData.getNumDimensions() );
    grt_assert( data.getNumSamples() == grtData.getNumSamples() );
    //Check the values match
    for(unsigned int i=0; i<data.getNumSamples(); i++){
        //Verify the class label
        grt_assert( data[i].getClassLabel() == grtData[i].getClassLabel() );
        //Verify the sample data
        for(unsigned int j=0; j<data.getNumDimensions(); j++){
            grt_assert( fabs( data[i][j] - grtData[i][j] ) < 1.0e-5 );
        }
    }
    cout << "grt formatted classification data OK\n";
    //Create a third classification data structure to load the CSV formatted data
    ClassificationData csvData;
    //Load the data from the GRT file
    csvData.load( "classification_data.csv" );
    //Check to make sure the sizes match
    grt_assert( data.getNumDimensions() == csvData.getNumDimensions() );
    grt_assert( data.getNumSamples() == csvData.getNumSamples() );
    //Check the values match
    for(unsigned int i=0; i<data.getNumSamples(); i++){
        //Verify the class label
        grt_assert( data[i].getClassLabel() == csvData[i].getClassLabel() );
        //Verify the sample data
        for(unsigned int j=0; j<data.getNumDimensions(); j++){
            grt_assert( fabs( data[i][j] - csvData[i][j] ) < 1.0e-5 );
        }
    }
    cout << "csv formatted classification data OK\n";
}

