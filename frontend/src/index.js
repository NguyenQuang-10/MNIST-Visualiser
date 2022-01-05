import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';



// generate array from 0 to n, n = 5 => [0,1,2,3,4], like python's range()
function range(n){
    return Array.from(Array(n).keys())
}

var MOUSE_DOWN = false
document.body.onmousedown = function() { 
    MOUSE_DOWN = true;
}
document.body.onmouseup = function() {
    MOUSE_DOWN = false;
}

class Dot extends React.Component {
    render() {
        return (
            <div className="dotOutline" onMouseOver={this.props.mouseOver}>
                <div className="dotInner" style={{opacity: this.props.activation}}></div>
            </div>
        );
    }

}

class Board extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            boardActivation: (new Array(28)).fill().map(function(){ return new Array(28).fill(0);}),
        }
        this.tempActivation = this.state.boardActivation
        this.updateDot = this.updateDot.bind(this);
    }

    updateDot(row, col){
        if (MOUSE_DOWN){
            console.log("doggo");
            this.tempActivation[row][col] = 1;
            this.setState({boardActivation: this.tempActivation});
        }

    }

    render() {
        return (
            <div className="Board">{
                    range(28).map( (row) => {
                            return (
                                <div key={row} className="board-row">
                                    {range(28).map( (col) => {
                                        return (
                                            <Dot key={col} activation={this.state.boardActivation[row][col]} mouseOver={() => this.updateDot(row, col)}></Dot>
                                        )
                                    }
                                    )}
                                </div>
                            )})
                }
            </div>
        );
    }


}

ReactDOM.render(<Board />, document.getElementById("root"));
