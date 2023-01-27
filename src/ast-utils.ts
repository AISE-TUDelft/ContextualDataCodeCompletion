import {Node} from "typescript";

/**
 * Reduce function for AST nodes. Will go down to all descendents
 * @param node Node to start at
 * @param fn Reduce function
 * @param initial Initial value
 */
export function reduceNode<T>(node: Node, fn: (acc: T, node: Node) => T, initial: T): T {
    let cur = fn(initial, node)
    node.forEachChild(child => {
        cur = reduceNode(child, fn, cur)
    })
    return cur
}