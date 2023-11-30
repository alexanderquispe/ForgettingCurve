
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import math

min_hl = 15.0 / ( 24 * 60 )
max_hl = 274.


class HLR_model:
    
    
    def __init__( self, feature_columns, omit_h_term = False, lrate = .001, alpha_ = .01, lambda_ = .1, sigma_ = 1. ):
        
        self.feature_columns = feature_columns
        self.omit_h_term     = omit_h_term
        self.theta           = defaultdict( float )
        self.fcounts         = defaultdict( int )
        self.lrate           = lrate
        self.alpha_          = alpha_
        self.lambda_         = lambda_
        self.sigma_          = sigma_
        
    
    def _find_h( self, row ):
        
        try: 
        
            dp = sum( self.theta[ index ] * row[ feature_name ] for index, feature_name in enumerate( self.feature_columns ) )
            return np.clip( 2**dp, min_hl, max_hl )
        
        except:
            
            return max_hl
    
    
    def _find_p( self, h_hat, delta_ ):
        
        p_hat = 2. ** ( -delta_ / h_hat )
        p_hat = np.clip( p_hat, 0.0001, .9999 )
        
        return p_hat
    
    
    def _estimate_losses( self, row, delta_, p , h ):

        h_hat = self._find_h( row )
        p_hat = self._find_p( h_hat, delta_ )
        slp   = ( p - p_hat ) ** 2
        slh   = ( h - h_hat ) ** 2
        
        return slp, slh
    
    
    def train_update( self, row ):
        
        X      = row[ self.feature_columns ].values.reshape( 1, -1 )
        p      = row[ 'p' ]
        delta_ = row[ 't' ]
        h      = row[ 'h' ]
        h_hat  = self._find_h( row )
        p_hat  = self._find_p( h_hat, delta_ )
        
        dlp_dw = 2. * ( p_hat - p ) * ( np.log( 2 ) ** 2 ) * p_hat * ( delta_ / h_hat )
        dlh_dw = 2. * ( h_hat - h ) * np.log( 2 ) * h_hat
        
        for index, feature_name in enumerate ( self.feature_columns ):
            
            feature_value          = row[ feature_name ]
            rate                   = ( 1 /( 1 + p ) ) * self.lrate / np.sqrt( 1 + self.fcounts[ index ] )
            # rate                   = self.lrate / np.sqrt( 1 + self.fcounts[ index ] )
            self.theta[ index ]   -= rate * dlp_dw * feature_value
            
            if not self.omit_h_term:
                
                self.theta[ index ] -= rate * self.alpha_ * dlh_dw * feature_value
                
            self.theta[ index ]   -= rate * self.lambda_ * self.theta[ index ] / self.sigma_ ** 2
            self.fcounts[ index ] += 1
            
            
    def train( self, trainset ):
        
        for i, row in trainset.iterrows():
            
            self.train_update( row )
        
            

    def test_model( self, testset ):
        
        results = { 'h' : [], 'p': [], 'h_hat': [], 'p_hat': [], 'slp': [], 'slh': []  }
        
        for i, row in testset.iterrows():             
        
            X        = row[ self.feature_columns ].values.reshape( 1, -1 )
            p        = row[ 'p' ]
            delta_   = row[ 't' ]
            h        = row[ 'h' ]
            h_hat    = self._find_h( row )
            p_hat    = self._find_p( h_hat, delta_ )
            slp, slh = self._estimate_losses( row, delta_, p, h )

            results[ 'h' ].append( h )
            results[ 'p' ].append( p )
            results[ 'h_hat' ].append( h_hat )
            results[ 'p_hat' ].append( p_hat )
            results[ 'slp' ].append( slp )
            results[ 'slh' ].append( slh )  
            
        mae_p      = mae( results[ 'p' ], results[ 'p_hat' ] )
        mae_h      = mae( results[ 'h' ], results[ 'h_hat' ] )
        cor_p      = spearman( results[ 'p' ], results[ 'p_hat' ] )
        cor_h      = spearman( results[ 'h' ], results[ 'h_hat' ] )
        total_slp  = sum( results[ 'slp' ] )
        total_slh  = sum( results[ 'slh' ] )
        total_l2   = sum( [ x ** 2 for x in self.theta.values() ] )
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
        
        with open( fname, 'w' ) as f:
            for ( k, v ), feature_name in zip( self.theta.items(), self.feature_columns ):
                f.write('%s\t%.4f\n' % ( feature_name, v ) )
                
                
                

class logit_model:
    
    
    def __init__( self, feature_columns, lrate = .001, alpha_ = .01, lambda_ = .1, sigma_ = 1. ):
        
        self.feature_columns = feature_columns
        self.theta           = defaultdict( float )
        self.fcounts         = defaultdict( int )
        self.lrate           = lrate
        self.alpha_          = alpha_
        self.lambda_         = lambda_
        self.sigma_          = sigma_
        
    
    def _find_h( self, h_seed_ ):
        
        return h_seed_.random()
    
    
    def _find_p( self, row ):
        
        dp    = sum( self.theta[ index ] * row[ feature_name ] for index, feature_name in enumerate( self.feature_columns ) )
        p_hat = np.clip( 1. / ( 1 + np.exp( -dp ) ),  0.0001, .9999 )
        
        return p_hat
    
    
    def _predict( self, row, p , h, h_seed_ ):

        h_hat = self._find_h( h_seed_ )
        p_hat = self._find_p( row )
        slp   = ( p - p_hat ) ** 2
        slh   = ( h - h_hat ) ** 2
        
        return p_hat, h_hat, slp, slh
    
    
    def train_update( self, row ):
        
        p     = row[ 'p' ]
        p_hat = self._find_p( row )
        error = p_hat - p

        for index, feature_name in enumerate ( self.feature_columns ):

            feature_value          = row[ feature_name ]
            rate                   = self.lrate / np.sqrt( 1 + self.fcounts[ index ] )
            self.theta[ index ]   -= rate * error * feature_value
            self.theta[ index ]   -= rate * self.lambda_ * self.theta[ index ] / self.sigma_ ** 2
            self.fcounts[ index ] += 1    
            
            
    def train( self, trainset ):
        
        for i, row in trainset.iterrows():
            
            self.train_update( row )
        
            

    def test_model( self, testset, h_seed = 2023 ):
        
        results = { 'h' : [], 'p': [], 'h_hat': [], 'p_hat': [], 'slp': [], 'slh': []  }
        h_seed_ = np.random.RandomState( h_seed )
        
        for i, row in testset.iterrows():             
        
            X        = row[ self.feature_columns ].values.reshape( 1, -1 )
            p        = row[ 'p' ]
            h        = row[ 'h' ]
            # h_hat    = self._find_h( h_seed_ )
            # p_hat    = self._find_p( row )
            p_hat, h_hat, slp, slh = self._predict( row, p, h, h_seed_ )

            results[ 'h' ].append( h )
            results[ 'p' ].append( p )
            results[ 'h_hat' ].append( h_hat )
            results[ 'p_hat' ].append( p_hat )
            results[ 'slp' ].append( slp )
            results[ 'slh' ].append( slh )
            
        mae_p      = mae( results[ 'p' ], results[ 'p_hat' ] )
        mae_h      = mae( results[ 'h' ], results[ 'h_hat' ] )
        cor_p      = spearman( results[ 'p' ], results[ 'p_hat' ] )
        cor_h      = spearman( results[ 'h' ], results[ 'h_hat' ] )
        total_slp  = sum( results[ 'slp' ] )
        total_slh  = sum( results[ 'slh' ] )
        total_l2   = sum( [ x ** 2 for x in self.theta.values() ] )
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
        
        with open( fname, 'w' ) as f:
            for ( k, v ), feature_name in zip( self.theta.items(), self.feature_columns ):
                f.write('%s\t%.4f\n' % ( feature_name, v ) )                
                
                
                
                
            
            
    def dump_theta( self, fname ):
        
        with open( fname, 'w' ) as f:
            for ( k, v ), feature_name in zip( self.theta.items(), self.feature_columns ):
                f.write('%s\t%.4f\n' % ( feature_name, v ) )                

                
                
            
def mae( l1, l2 ):

    mae = np.mean( [ abs( l1 [ i ] - l2[ i ] ) for i in range(len( l1 ) ) ] )

    return round( mae, 3 )


def spearman( l1, l2 ):

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
                          
                          
def read_data( df, method, omit_lexemes = False  ):
    
    df[ 'p' ]          = df[ 'p_recall' ].apply( lambda x: np.clip( float( x ),  0.0001, .9999 ) )
    df[ 't' ]          = df[ 'delta' ].apply( lambda x: float( x ) / ( 60 * 60 * 24 ) )
    df[ 'h' ]          = df.apply( lambda row: np.clip( -row[ 't' ] / np.log2( row [ 'p' ] ), min_hl, max_hl ), axis = 1 )
    df[ 'lang' ]       = df.apply( lambda row: f"{ row[ 'ui_language' ] } -> { row[ 'learning_language' ] }", axis = 1 )
    df[ 'lexeme' ]     = df.apply( lambda row: f"{ row[ 'learning_language' ] }:{ row[ 'lexeme_string' ] }", axis = 1 )
    df[ 'right' ]      = df[ 'history_correct' ].astype( int )
    df[ 'wrong' ]      = df[ 'history_seen' ].astype( int ) - df[ 'right' ]    
    df[ 'right_this' ] = df[ 'session_correct' ].astype( int )
    df[ 'wrong_this' ] = df[ 'session_seen' ].astype( int ) - df[ 'right_this' ]  
    df[ 'right' ]      = df[ 'right' ].apply( lambda x: np.sqrt( 1 + x ) )
    df[ 'wrong' ]      = df[ 'wrong' ].apply( lambda x: np.sqrt( 1 + x ) )
    df[ 'bias' ]       = 1
    df[ 'time' ]       = df[ 't' ] if method == 'lr' else None
                         
    if not omit_lexemes:
    
        lexeme_dummies = pd.get_dummies( df[ 'lexeme' ], dtype = float )
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat( [ df, lexeme_dummies ], axis = 1 )
        
        feature_vars = [ 'right', 'wrong', 'bias' ] + ( [ 'time' ] if method == 'lr' else [] ) + lexeme_columns
                         
    else:
                         
        feature_vars = [ 'right', 'wrong', 'bias' ] + ( [ 'time' ] if method == 'lr' else [] )   
    
    splitpoint = int(0.9 * len(df))
    trainset   = df[ : splitpoint ]
    testset    = df[ splitpoint : ]

    return trainset, testset, feature_vars


