(function(exports) {
  var SVM = function(options) {
    this.options = options || {};
    this.C = options.C || 1.0;
    this.tol = options.tol || 1e-5;
    this.sigma = options.sigma || 0.5;

    if ( options.kernel === undefined || options.kernel === "linear") {
      this.kernel = this._linear;
    } else if (options.kernel === "rbf") {
      this.kernel = this._rbf;
    } else {
      throw new Error().stack;
    }

  };

  SVM.prototype = {
    train: function(samples, labels, maxIterate) {
      var i,j;

      var Ei, Ej;
      var ai, aj, ai_, aj_;
      var b1, b2;
      var L, H;
      var eta;

      var iter = 0;
      var passes = 0;
      var alphaChaned = 0;

      // 检测参数
      if ( maxIterate === undefined) {
        maxIterate = 500;
      }
      if (samples.length != labels.length) {
        throw new Error().stack;
      }

      // 加载数据
      this.x = [];
      this.y = [];
      this.alphas = [];
      this.b = 0.0;
      for (i = 0; i < samples.length; i++) {
          this.alphas.push(0.0);
          this.y.push( labels[i][0] === 1 ? 1 : -1 );
          this.x.push( samples[i] );
      }

      // run SMO algorithm
      while(passes < 10 && iter < maxIterate) {
          alphaChaned = 0;
          for( i = 0; i < this.alphas.length; i++) {
              Ei = (this.pred(this.x[i]) - this.y[i]) * this.y[i];
              if ( (Ei < -1 * this.tol && this.alphas[i] < this.C)
                || (Ei > this.tol && this.alphas[i] > 0) ) {

                  Ei = Ei * this.y[i];

                  j = i;
                  while(j === i) {
                      j = Math.floor(Math.random()*this.alphas.length); // 0 ~ N
                  }
                  Ej = (this.pred(this.x[j]) - this.y[j]);

                  ai = this.alphas[i];
                  aj = this.alphas[j];

                  if(this.y[i] === this.y[j]) {
                      L = Math.max(0, ai + aj - this.C);
                      H = Math.min(this.C , ai + aj);
                  } else {
                      L = Math.max(0, aj - ai);
                      H = Math.min(this.C, this.C + aj - ai);
                  }
                  if(Math.abs(L - H) < 1e-4) {
                       continue;
                  }

                  eta = 2 * this.kernel(this.x[i], this.x[j]) - this.kernel(this.x[i], this.x[i]) - this.kernel(this.x[j], this.x[j]);
                  if(eta >= 0) {
                      continue;
                  }

                  // update ai and aj
                  aj_ = aj - this.y[j] * (Ei - Ej) / eta;
                  if ( aj_  > H ) {
                      aj_ = H;
                  }
                  if ( aj_ < L ) {
                      aj_ = L;
                  }
                  if(Math.abs(aj - aj_) < 1e-4) {
                      continue;
                  }
                  this.alphas[j] = aj_;
                  ai_ = ai + this.y[i] * this.y[j] * ( aj - aj_);
                  this.alphas[i] = ai_;

                  //update b
                  b1 = this.b - Ei - this.y[i]*(ai_ - ai)*this.kernel(this.x[i], this.x[i])
                           - this.y[j]*(aj_ - aj)*this.kernel(this.x[i], this.x[j]);

                  b2 = this.b - Ej - this.y[j]*(aj_ - aj)*this.kernel(this.x[i], this.x[j])
                           - this.y[j]*(aj_ - aj)*this.kernel(this.x[j], this.x[j]);

                  this.b = 0.5*(b1+b2);
                  if ( ai_ > 0 && ai_ < this.C) {
                      this.b = b1;
                  }
                  if ( aj_ > 0 && aj_ < this.C) {
                      this.b = b2;
                  }
                  alphaChaned++;
              }
          }

          iter++;
          if(alphaChaned == 0) {
              passes++;
          } else {
              passes= 0;
          }
      }
    },

    pred : function(x) {
        var ret = 0.0;
        for(var i=0; i < this.alphas.length; i++) {
            ret += this.alphas[i] * this.y[i] * this.kernel(x, this.x[i]);
        }
        ret += this.b;
        return ret;
    },

    _rbf : function(v1, v2) {
        var s=0;
        for(var q=0;q<v1.length;q++) { s += (v1[q] - v2[q])*(v1[q] - v2[q]); }
        return Math.exp(-s/(2.0*this.sigma*this.sigma));
    },

    _linear : function(x1, x2) {
        var i;
        var sum = 0.0;
        for(i = 0; i < x1.length; i++) {
          sum = sum + x1[i] * x2[i];
        }
        return sum;
    },

  };

  exports.SVM = SVM;

})( (typeof module != 'undefined' && module.exports) || ekj );
