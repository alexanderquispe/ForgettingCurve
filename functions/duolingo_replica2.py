
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import math

min_hl = 15.0 / ( 24 * 60 )
max_hl = 274.


class HLR_model:
    
    
    def __init__( self, feature_columns, lrate = .001, alpha_ = .01, lambda_ = 0.1, sigma_ = 1. ):
        
        self.feature_columns = feature_columns
        self.theta         = np.random.randn( len( feature_columns ) ) * 0.01
        self.fcounts         = defaultdict( float )
        self.lrate           = lrate
        self.alpha_          = alpha_
        self.lambda_         = lambda_
        self.sigma_          = sigma_
        
    
    def _find_h( self, X ):
        
        h_hat = 2 ** ( X.dot( self.theta ) )
        h_hat = np.clip( h_hat, min_hl, max_hl )
        
        return h_hat
    
    
    def _find_p( self, h_hat, delta_ ):
        
        p_hat = 2 ** ( -delta_ / h_hat )
        p_hat = np.clip( p_hat, min_hl, max_hl )
        
        return p_hat
    
    
    def _estimate_losses( self, X, delta_, p , h ):

        h_hat = self._find_h( X )
        p_hat = self._find_p( h_hat, delta_ )
        slp   = ( p - p_hat ) ** 2
        slh   = ( h - h_hat ) ** 2
        
        return slp, slh
        
    
    def _estimate_total_losses( self, X, delta_, p, h ):
        
        h_hat      = self._find_h( X )
        p_hat      = self._find_p( h_hat, delta_ )
        slp, slh   = self._estimate_losses( X, delta_, p, h )
        total_slp  = np.sum( slp )
        total_slh  = np.sum( slh )
        total_l2   = np.sum( self.theta ** 2 )
        total_loss = total_slp + self.alpha_ * total_slh + self.lambda_ * total_l2
        
        return total_slp, total_slh, total_loss, total_loss
    
    
    def train_update( self, row ):
        
        X      = row[ self.feature_columns ].values.reshape( 1, -1 )
        p      = row[ 'p' ]
        delta_ = row[ 't' ]
        h      = row[ 'h' ]
        h_hat  = self._find_h( X )
        p_hat  = self._find_p( h_hat, delta_ )
        
        dlp_dw = 2. * ( p_hat - p ) * ( np.log( 2 ) ** 2 ) * p_hat * ( delta_ / h )
        dlh_dw = 2. * ( h_hat - h ) * np.log( 2 ) * h
        
        for index, feature_name in enumerate ( self.feature_columns ):
            feature_value          = row[ feature_name ]
            rate                   = ( 1 /( 1 + p ) ) * self.lrate / np.sqrt( 1 + self.fcounts[ index ] )
            self.theta[ index ]   -= rate * self.alpha_ * dlp_dw * feature_value
            self.theta[ index ]   -= rate * self.alpha_ * self.theta[ index ] / self.sigma_ ** 2
            self.fcounts[ index ] += 1
            
            
    def train( self, trainset ):
        
        for i, row in trainset.iterrows():
            
            self.train_update( row )
        
            

    def test_model( self, testset ):
        
        results = { 'h' : [], 'p': [], 'h_hat': [], 'p_hat': [], 'total_slp': [], 'total_slh': []  }
        
        for i, row in testset.iterrows():             
        
            X      = row[ self.feature_columns ].values.reshape( 1, -1 )
            p      = row[ 'p' ]
            delta_ = row[ 't' ]
            h      = row[ 'h' ]
            h_hat  = self._find_h( X )
            p_hat  = self._find_p( h_hat, delta_ )
            total_slp, total_slh, total_loss, total_loss = self._estimate_total_losses( X, delta_, p , h )

            results[ 'h' ].append( h )
            results[ 'p' ].append( p )
            results[ 'h_hat' ].append( h_hat )
            results[ 'p_hat' ].append( p_hat )
            results[ 'total_slp' ].append( total_slp )
            results[ 'total_slh' ].append( total_slh )
            
        mae_p      = round( mae( results[ 'p' ], results[ 'p_hat' ] ), 3 )
        mae_h      = round( mae( results[ 'h' ], results[ 'h_hat' ] ), 3 )
        cor_p      = spearman( results[ 'p' ], results[ 'p_hat' ] )
        cor_h      = spearman( results[ 'h' ], results[ 'h_hat' ] )
        total_slp  = sum( results[ 'total_slp' ] )
        total_slh  = round( sum( results[ 'total_slh' ] ), 3 )
        total_l2   = sum( [ x ** 2 for x in self.theta ] )
        total_loss = round( ( total_slp + self.alpha_ * total_slh + self.lambda_ * total_l2 ), 3 )
        
        print( '-----------------------------------'                       )
        print(          'Results'                                          )
        print( '-----------------------------------'                       ) 
        print( f'Total Loss : { total_loss }'                              )
        print( f'p          : { total_slp }'                               )
        print( f'h          : { round( ( self.alpha_ * total_slh ), 3 ) }' )
        print( f'l2         : { round( ( self.lambda_ * total_l2 ), 3) }'  )
        print( f'mae (p)    : { mae_p }'                                   )
        print( f'cor (p)    : { cor_p }'                                   )
        print( f'mae (h)    : { mae_h }'                                   )
        print( f'cor (h)    : { cor_h }'                                   )
        print( '-----------------------------------'                       )
            
            
def mae( l1, l2 ):

    mae = np.mean( [ abs( l1 [ i ] - l2[ i ] ) for i in range(len( l1 ) ) ] )

    return round( mae, 3 )


def spearman( l1, l2 ):

    m1  = float( np.sum( l1 ) )/len( l1 )
    m2  = float( np.sum( l2 ) )/len( l2 )
    num = 0.
    d1  = 0.
    d2  = 0.
    
    for i in range(len( l1 ) ):
        num += ( l1[ i ] - m1 ) * ( l2[ i ] - m2 )
        d1  += ( l1[ i ] - m1 ) ** 2
        d2  += ( l2[ i ] - m2 ) ** 2
        
    return np.mean( num / math.sqrt( d1 * d2 ) )



# def pclip(p):
#     # ... definición de pclip ...
#     return min(max(p, 0.0001), .9999)

# def hclip(h):
#     # ... definición de hclip ...
#     MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes
#     MAX_HALF_LIFE = 274.              # 9 months
#     return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)

                          
                          
def read_data( df, seed = 45, omit_lexemes = False,  ):
    
    df[ 'p' ]          = df[ 'p_recall' ].apply( lambda x: np.clip( float( x ),  0.0001, .9999 ) )
    df[ 't' ]          = df[ 'delta' ].apply( lambda x: float( x ) / ( 60 * 60 * 24 ) )
    df[ 'h' ]          = df.apply( lambda row: np.clip( -row[ 't' ] / np.log2( row [ 'p' ] ), min_hl, max_hl ), axis = 1 )
    df[ 'lang' ]       = df.apply( lambda row: f"{ row[ 'ui_language' ] } -> { row[ 'learning_language' ] }", axis = 1 )
    df[ 'lexeme' ]     = df[ 'learning_language' ] + ':' + df[ 'lexeme_string' ]
    df[ 'right' ]      = df[ 'history_correct' ].astype( int )
    df[ 'wrong' ]      = df[ 'history_seen' ].astype( int ) - df[ 'right' ]    
    df[ 'right_this' ] = df[ 'session_correct' ].astype( int )
    df[ 'wrong_this' ] = df[ 'session_seen' ].astype( int ) - df[ 'right_this' ]  
    df[ 'right' ]      = df[ 'right' ].apply( lambda x: np.sqrt( 1 + x ) )
    df[ 'wrong' ]      = df[ 'wrong' ].apply (lambda x: np.sqrt( 1 + x ) )
    df[ 'bias' ]       = 1
                         
    if not omit_lexemes:
    
        lexeme_dummies = pd.get_dummies( df[ 'lexeme' ], dtype = float )
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat( [ df, lexeme_dummies ], axis = 1 )
        
        feature_vars = [ 'right', 'wrong', 'bias' ] + lexeme_columns
                         
    else:
                         
        feature_vars = [ 'right', 'wrong', 'bias' ]
                         
    np.random.seed( seed )                     
    msk      = np.random.rand( len( df ) ) < 0.9
    trainset = df[ msk ]
    testset  = df[ ~msk ]                         

    return trainset, testset, feature_vars
