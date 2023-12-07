
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
            
            theta_array = np.array([self.theta[feature_name] for feature_name in self.feature_columns])
            row_array = row[self.feature_columns].values

            dp = np.dot(theta_array, row_array)
    
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
                          
                          
def read_data(df, omit_lexemes=False):
    # Usando operaciones vectorizadas para 'p' y 't'
    df['p'] = np.clip(df['p_recall'].astype(float), 0.0001, .9999)
    df['t'] = df['delta'].astype(float) / (60 * 60 * 24)

    # Calculando 'h' de manera vectorizada
    df['h'] = np.clip(-df['t'] / np.log2(df['p']), min_hl, max_hl)

    # Concatenando cadenas de manera vectorizada
    df['lang'] = df['ui_language'] + " -> " + df['learning_language']
    df['lexeme'] = df['learning_language'] + ":" + df['lexeme_string']

    # Convirtiendo y realizando operaciones aritméticas de manera vectorizada
    df['right'] = np.sqrt(1 + df['history_correct'].astype(int))
    df['wrong'] = np.sqrt(1 + df['history_seen'].astype(int) - df['right'])
    df['right_this'] = df['session_correct'].astype(int)
    df['wrong_this'] = df['session_seen'].astype(int) - df['right_this']

    # Añadiendo una columna de sesgo
    df['bias'] = 1

    # Tratamiento de lexemas (si es necesario)
    if not omit_lexemes:
        lexeme_dummies = pd.get_dummies(df['lexeme'], dtype=float)
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat([df, lexeme_dummies], axis=1)
        feature_vars = ['right', 'wrong', 'bias'] + lexeme_columns
    else:
        feature_vars = ['right', 'wrong', 'bias']

    # Dividiendo el conjunto de datos en entrenamiento y prueba
    splitpoint = int(0.9 * len(df))
    trainset = df[:splitpoint]
    testset = df[splitpoint:]

    return trainset, testset, feature_vars


