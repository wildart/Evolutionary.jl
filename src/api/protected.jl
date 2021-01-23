# Collection of protected function for GP

"""Protected division"""
pdiv(x, y, undef=10e6) = ifelse(y==0 , x+undef  , div(x,y))
"""Protected exponential"""
pexp(x, undef=10e15)   = ifelse(x>=32, x+undef  , exp(x))
"""Protected natural log"""
plog(x, undef=10e6)    = ifelse(x==0 , -undef   , log(abs(x)))
"""Protected sq.root"""
psqrt(x)               = sqrt(abs(x))
"""Protected sin(x)"""
psin(x, undef=10e6)    = ifelse(isinf(x), undef, sin(x))
"""Protected cos(x)"""
pcos(x, undef=10e6)    = ifelse(isinf(x), undef, cos(x))
"""Protected exponentiation operation"""
function ppow(x, y, undef=10e6)
    if y>=10
        x+y+undef
    elseif y < 1
        pow(abs(x),y)
    else
        pow(x,y)
    end
end
