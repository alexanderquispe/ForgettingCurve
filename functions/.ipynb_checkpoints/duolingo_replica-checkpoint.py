
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import math

min_hl = 15.0 / ( 24 * 60 )
max_hl = 274.


class HLR_model:
    
    def __init__( self, feature_columns, omit_h_term = False, lrate = .001, alpha_ = .01, lambda_ = .1, sigma_ = 1. ):
        
        """
        Initialize the Half-Life Regression (HLR) model.
        
        Input:
        - feature_columns : List of feature names used in the model.
        - omit_h_term     : Boolean flag to omit the half-life related term in weight updates.
        - lrate           : Learning rate for model training.
        - alpha_          : Weight for the half-life loss term.
        - lambda_         : Weight for L2 regularization.
        - sigma_          : Parameter for L2 regularization.
        """
        
        self.feature_columns = feature_columns
        self.omit_h_term     = omit_h_term
        self.theta           = np.zeros( len( feature_columns ) )
        self.fcounts         = np.zeros( len( feature_columns ) )
        self.lrate           = lrate
        self.alpha_          = alpha_
        self.lambda_         = lambda_
        self.sigma_          = sigma_

    def _find_h( self, X ):
        
        """
        Calculate the estimated half-life for a given input.
        
        Input:
        - X: Feature values for which to calculate the half-life.
        
        Output:
        - Estimated half-life values.
        """
        
        dp = np.dot( X, self.theta )
        return np.clip( 2 ** dp, min_hl, max_hl )

    
    def _find_p( self, h_hat, delta_ ):
        
        """
        Calculate the predicted probability based on the estimated half-life and time delta.
        
        Input:
        - h_hat  : Estimated half-life.
        - delta_ : Time delta since the last recall event.
        
        Output:
        - Predicted probability of recall.
        """
        
        p_hat = 2 ** ( -delta_ / h_hat )
        return np.clip( p_hat, 0.0001, .9999 )
    
  

    
    def _estimate_losses( self, X, delta_, p, h ):
        
        """
        Estimate the squared loss for probability and half-life.
        
        Input:
        - X      : Feature values.
        - delta_ : Time delta.
        - p      : Actual probability of recall.
        - h      : Actual half-life.
        
        Output:
        - Squared loss for probability and half-life.
        """
        
        h_hat = self._find_h( X )
        p_hat = self._find_p( h_hat, delta_ )
        slp   = ( p - p_hat ) ** 2
        slh   = ( h - h_hat ) ** 2
        return slp, slh

    
    def train_update( self, X, p, delta_, h ):
        
        """
        Perform a single update of the model weights using one training example.
        
        Input:
        - X      : Feature values of the training example.
        - p      : Actual probability of recall for the training example.
        - delta_ : Time delta for the training example.
        - h      : Actual half-life for the training example.
        """
        
        h_hat  = self._find_h( X )
        p_hat  = self._find_p( h_hat, delta_ )

        dlp_dw = 2. * ( p_hat - p ) * ( np.log( 2 ) ** 2 ) * p_hat * ( delta_ / h_hat )
        dlh_dw = 2. * ( h_hat - h ) * np.log( 2 ) * h_hat

        rate   = ( 1 / ( 1 + p ) ) * self.lrate / np.sqrt( 1 + self.fcounts )

        self.theta -= rate * dlp_dw * X

        # True when self.omit_h_term is False
        if not self.omit_h_term:
            self.theta -= rate * self.alpha_ * dlh_dw * X

        self.theta   -= rate * self.lambda_ * self.theta / self.sigma_ ** 2
        self.fcounts += 1
        

    def train( self, trainset ):

        """
        Train the HLR model using a set of training data.
        
        Input:
        - trainset: Training dataset containing features and target variables.
        """
        
        X      = trainset[ self.feature_columns ].values
        p      = trainset[ 'p' ].values
        delta_ = trainset[ 't' ].values
        h      = trainset[ 'h' ].values

        for i in range( len( trainset ) ):
            self.train_update( X[ i ], p[ i ], delta_[ i ], h[ i ] )
            
            
            
    def test_model( self, testset ):
        
        """
        Test the HLR model on a given dataset and print out performance metrics.
        
        Input:
        - testset: Test dataset.
        """
        
        results = { 'h': [], 'p': [], 'h_hat': [], 'p_hat': [], 'slp': [], 'slh': [] }
        
        X      = testset[ self.feature_columns ].values
        p      = testset[ 'p' ].values
        delta_ = testset[ 't' ].values
        h      = testset[ 'h' ].values

        h_hat    = self._find_h( X )
        p_hat    = self._find_p( h_hat, delta_ )
        slp, slh = self._estimate_losses( X, delta_, p, h )

        results[ 'h' ]     = h.tolist()
        results[ 'p' ]     = p.tolist()
        results[ 'h_hat' ] = h_hat.tolist()
        results[ 'p_hat' ] = p_hat.tolist()
        results[ 'slp' ]   = slp.tolist()
        results[ 'slh' ]   = slh.tolist()

        mae_p      = mae( results[ 'p' ], results[ 'p_hat' ] )
        mae_h      = mae( results[ 'h' ], results[ 'h_hat' ] )
        cor_p      = spearman( results[ 'p' ], results[ 'p_hat' ] )
        cor_h      = spearman( results[ 'h' ], results[ 'h_hat' ] )
        total_slp  = sum( results[ 'slp' ] )
        total_slh  = sum( results[ 'slh' ] )
        total_l2   = sum( self.theta ** 2 )
        total_loss = total_slp + self.alpha_ * total_slh + self.lambda_ * total_l2
        
        print( '-----------------------------'                )
        print( '            Results          '                )
        print( '-----------------------------'                ) 
        print( f'Total Loss : { total_loss:.3f}'              )
        print( f'p          : { total_slp:.3f}'               )
        print( f'h          : { self.alpha_ * total_slh:.3f}' )
        print( f'l2         : { self.lambda_ * total_l2:.3f}' )
        print( f'mae (p)    : { mae_p:.3f}'                   )
        print( f'cor (p)    : { cor_p:.3f}'                   )
        print( f'mae (h)    : { mae_h:.3f}'                   )
        print( f'cor (h)    : { cor_h:.3f}'                   )
        print( '-----------------------------'                )
        
                    
            
    def dump_theta( self, fname ):

        """
        Output the model weights to a file.
        
        Input:
        - fname: Filename where the model weights will be saved.
        """
        
        with open( fname, 'w' ) as f:
            
            for index, value in enumerate( self.theta ):
                feature_name = self.feature_columns[ index ]
                f.write( f'{ feature_name }\t{ value:.4f}\n' )
                
                
                

class logit_model:
    
    
    def __init__( self, feature_columns, lrate = .001, alpha_ = .01, lambda_ = .1, sigma_ = 1. ):
        
        """
        Initialize the Logistic Regression model.

        Inputs:
        - feature_columns : List of feature names used in the model.
        - lrate           : Learning rate for model training.
        - alpha_          : Weight for the loss term related to h.
        - lambda_         : Weight for L2 regularization.
        - sigma_          : Parameter for L2 regularization.
        """
        
        self.feature_columns = feature_columns
        self.theta           = np.zeros( len( feature_columns ) )
        self.fcounts         = np.zeros( len( feature_columns ) )
        self.lrate           = lrate
        self.alpha_          = alpha_
        self.lambda_         = lambda_
        self.sigma_          = sigma_
        
    
    def _find_h( self, h_seed_, size ):
        
        """
        Generate random values for h, used in model testing.

        Inputs:
        - h_seed_ : Random seed for reproducibility.
        - size    : Number of random values to generate.

        Output:
        - Array of random values for h.
        """
        
        return h_seed_.random( size = size )
    
    
    def _find_p( self, X ):
        
        """
        Calculate the predicted probability (p_hat).
        
        Input:
        - X: Feature values for prediction.

        Output:
        - p_hat: Predicted probabilities.
        """
        
        dp = np.dot( X, self.theta )      
        return np.clip( 1. / ( 1 + np.exp( -dp ) ),  0.0001, .9999 )
    
    
    def _predict( self, X, p , h, h_seed_ ):
        
        """
        Make predictions using the model and calculate the squared loss.

        Inputs:
        - X       : Feature values for prediction.
        - p       : Actual probability values.
        - h       : Actual h values (used for comparison).
        - h_seed_ : Random seed for generating h_hat values.

        Output:
        - p_hat : Predicted probabilities.
        - h_hat : Randomly generated h values.
        - slp   : Squared loss for p.
        - slh   : Squared loss for h.
        """

        h_hat = self._find_h( h_seed_, len( X ) )
        p_hat = self._find_p( X )
        slp   = ( p - p_hat ) ** 2
        slh   = ( h - h_hat ) ** 2
        
        return p_hat, h_hat, slp, slh
    
    
    
    
    def train_update( self, X, p ):
        
        """
        Update the model weights based on one training example.

        Inputs:
        - X: Feature values of the training example.
        - p: Actual probability of the training example.

        Output:
        - None. Model weights are updated in place.
        """
        
        p_hat = self._find_p( X )
        error = p_hat - p
        
        rate  = self.lrate / np.sqrt( 1 + self.fcounts )
        
        self.theta -= rate * error * X
        self.theta -= rate * self.lambda_ * self.theta / self.sigma_ ** 2
        self.fcounts += 1
        
        
            
            
    def train( self, trainset ):
        
        """
        Train the Logistic Regression model using a set of training data.

        Input:
        - trainset: Training dataset containing features and target variables.
        """
        
        X      = trainset[ self.feature_columns ].values
        p      = trainset[ 'p' ].values

        for i in range( len( trainset ) ):
            self.train_update( X[ i ], p[ i ] )
        
            

    def test_model( self, testset, h_seed = 2023 ):
        
        """
        Test the model on a given dataset and print out performance metrics.

        Inputs:
        - testset: Test dataset.
        - h_seed: Random seed for generating h_hat values in testing.
        """
        
        results = { 'h' : [], 'p': [], 'h_hat': [], 'p_hat': [], 'slp': [], 'slh': []  }
        h_seed_ = np.random.RandomState( h_seed )
        
        X      = testset[ self.feature_columns ].values
        p      = testset[ 'p' ].values
        h      = testset[ 'h' ].values
        p_hat, h_hat, slp, slh = self._predict( X, p, h, h_seed_ )
        
        results[ 'h' ]     = h.tolist()
        results[ 'p' ]     = p.tolist()
        results[ 'h_hat' ] = h_hat.tolist()
        results[ 'p_hat' ] = p_hat.tolist()
        results[ 'slp' ]   = slp.tolist()
        results[ 'slh' ]   = slh.tolist()
    
        mae_p      = mae(results[ 'p' ], results[ 'p_hat' ] )
        mae_h      = mae(results[ 'h' ], results[ 'h_hat' ] )
        cor_p      = spearman(results[ 'p' ], results[ 'p_hat' ] )
        cor_h      = spearman(results[ 'h' ], results[ 'h_hat' ] )
        total_slp  = sum(results[ 'slp' ] )
        total_slh  = sum(results[ 'slh' ] )
        total_l2   = sum( self.theta ** 2 )
        total_loss = total_slp + self.alpha_ * total_slh + self.lambda_ * total_l2
        
        print( '-----------------------------'                )
        print( '            Results          '                )
        print( '-----------------------------'                ) 
        print( f'Total Loss : { total_loss:.3f}'              )
        print( f'p          : { total_slp:.3f}'               )
        print( f'h          : { self.alpha_ * total_slh:.3f}' )
        print( f'l2         : { self.lambda_ * total_l2:.3f}' )
        print( f'mae (p)    : { mae_p:.3f}'                   )
        print( f'cor (p)    : { cor_p:.3f}'                   )
        print( f'mae (h)    : { mae_h:.3f}'                   )
        print( f'cor (h)    : { cor_h:.3f}'                   )
        print( '-----------------------------'                )
            
                
    def dump_theta( self, fname ):
        
        """
        Output the model weights to a file.

        Input:
        - fname: Filename where the model weights will be saved.
        """
        
        with open( fname, 'w' ) as f:
            
            for index, value in enumerate( self.theta ):
                feature_name = self.feature_columns[ index ]
                f.write( f'{ feature_name }\t{ value:.4f}\n' )
                
                
            
def mae( l1, l2 ):
    
    """
    Calculate the Mean Absolute Error (MAE) between two lists.

    Inputs:
    - l1: First list of actual values.
    - l2: Second list of predicted values.

    Output:
    - The MAE rounded to three decimal places.
    """

    mae = np.mean( [ abs( l1 [ i ] - l2[ i ] ) for i in range(len( l1 ) ) ] )

    return mae


def spearman( l1, l2 ):
    
    """
    Calculate the Spearman rank correlation coefficient between two lists.

    Inputs:
    - l1: First list of values.
    - l2: Second list of values.

    Output:
    - Spearman rank correlation coefficient.
    """

    m1  = float( np.sum( l1 ) ) / len( l1 )
    m2  = float( np.sum( l2 ) ) / len( l2 )
    num = 0.
    d1  = 0.
    d2  = 0.
    
    for i in range(len( l1 ) ):
        num += ( l1[ i ] - m1 ) * ( l2[ i ] - m2 )
        d1  += ( l1[ i ] - m1 ) ** 2
        d2  += ( l2[ i ] - m2 ) ** 2
        
        
    return num / math.sqrt( d1 * d2 )
                          


def read_data( df, method, omit_lexemes = False ):
    
    """
    Preprocess the input DataFrame for Half-Life Regression or Logistic Regression model.

    Inputs:
    - df           : Input DataFrame containing raw data.
    - method       : Model type ('hlr' for Half-Life Regression or 'lr' for Logistic Regression).
    - omit_lexemes : Boolean flag to omit lexeme features.

    Output:
    - trainset     : Processed training dataset.
    - testset      : Processed testing dataset.
    - feature_vars : List of feature variable names.
    """
    
    df[ 'p' ]          = np.clip( df[ 'p_recall' ].astype( float ), 0.0001, 0.9999 )
    df[ 't' ]          = df[ 'delta' ].astype( float ) / ( 60 * 60 * 24 )
    df[ 'h' ]          = np.clip( -df[ 't' ] / np.log2( df[ 'p' ] ), min_hl, max_hl )
    df[ 'lang' ]       = df[ 'ui_language' ] + '->' + df[ 'learning_language' ]
    df[ 'lexeme' ]     = df[ 'learning_language' ] + ':' + df[ 'lexeme_string' ]
    df[ 'right' ]      = df[ 'history_correct' ].astype( int )
    df[ 'wrong' ]      = df[ 'history_seen' ].astype( int ) - df[ 'right' ]    
    df[ 'right_this' ] = df[ 'session_correct' ].astype( int )
    df[ 'wrong_this' ] = df[ 'session_seen' ].astype( int ) - df[ 'right_this' ]     
    df[ 'right' ]      = np.sqrt( 1 + df[ 'right' ].astype( int ) )
    df[ 'wrong' ]      = np.sqrt( 1 + df[ 'wrong' ].astype( int ) )
    df[ 'bias' ]       = 1
    df[ 'time' ]       = df[ 't' ] if method == 'lr' else None

    if not omit_lexemes:
    
        lexeme_dummies = pd.get_dummies( df[ 'lexeme' ], dtype = float )
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat( [ df, lexeme_dummies ], axis = 1 )
        
        feature_vars = [ 'right', 'wrong' ] + ( [ 'time' ] if method == 'lr' else [] ) + [ 'bias' ] + lexeme_columns
                         
    else:
                         
        feature_vars = [ 'right', 'wrong' ] + ( [ 'time' ] if method == 'lr' else [] ) + [ 'bias' ]
    
    splitpoint = int( 0.9 * len( df ) )
    trainset   = df.iloc[ : splitpoint ]
    testset    = df.iloc[ splitpoint : ]

    return trainset, testset, feature_vars