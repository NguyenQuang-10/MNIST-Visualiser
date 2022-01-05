import React from 'react';
import ReactDOM from 'react-dom';
import './board.js';

// generate array from 0 to n, n = 5 => [0,1,2,3,4], like python's range()
function range(n){
    return Array.from(Array(n).keys())
}

class Dot extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            activation: this.props.activation,
        }
    }

    setActivation(a) {
        this.setState({activation: a});
    }

    render() {
        return (
            <div className="dotOutline">
                <div className="dotInner" style={{opacity = this.state.activation}}></div>
            </div>
        );
    }

}

class Board extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            boardActivation: new Array(784).fill(0),
        }
        this.getActivation = this.getActivation.bind(this);
        this.boardArray = this.state.boardActivation;
    }

    // may need to get fix
    getActivation(index, activation){
        this.boardArray[index] = activation;
        this.setState({boardActivation: this.boardArray});
    }

    render() {
        return (
            <div className="Board">{
                    range(28).map( (row) => {
                            return (
                                <div key={row} className="board-row">
                                    {range(28).map( (col) => {
                                        <Dot key={col} activation={0} ></Dot>   
                                    }
                                    )}
                                </div>
                            )
                        }

                    )
                }
            </div>
        );
    }


}