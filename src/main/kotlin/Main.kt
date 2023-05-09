import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import java.lang.Double

fun main(args: Array<String>) {

    val g = Graph()
    val ops = Ops.create(g)

    val a = ops.placeholder(Double::class.java)
    val b = ops.placeholder(Double::class.java)

    val mul = ops.math.mul(a, b)

    val tensorA = Tensor.create(3.0)
    val tensorB = Tensor.create(4.1)
    val session = Session(g)
    session.runner()
        .feed(a, tensorA)
        .feed(b, tensorB)
        .fetch(mul)
        .run().forEach {
            println(it)
        }
}

