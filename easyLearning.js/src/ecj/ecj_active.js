(function(exports) {
  "use strict";

  var TanhAct = {
    forward: function(x) {
      var y = Math.exp(2 * x);
      if (isFinite(y)) {
        return 1;
      }
      return (y - 1) / (y + 1);
    },
    backward: function(x,y) {
      return (1 - y*y);
    },
  };

  var SigmoidAct = {
    forward: function(x) {
      return 1 / ( 1 + Math.exp(-x) );
    },
    backward:function(x, y) {
      return y * ( 1 - y);
    },
  };

  var BypassAct = {
    forward: function(x) {
      return x;
    },
    backward: function(x,y) {
      return 1;
    },
  };

  var ReluAct = {
    forward: function(x) {
      return x;
    },
    backward: function(x,y) {
      if(y>0) {
        return 1;
      } else {
        return 0;
      }
    },
  };

  exports.act = {};
  exports.act.getActivation = function(actType) {
    if ( actType === "sigmoid" ) {
      return SigmoidAct;
    } else if ( actType === "bypass") {
      return BypassAct;
    } else if ( actType === "relu" ) {
      return ReluAct;
    } else if ( actType === "tanh" ) {
      return TanhAct;
    } else {
      throw new Error().stack;
    }
  }

})((typeof module != 'undefined' && module.exports) || ecj )
