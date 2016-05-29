#include "ofApp.h"
#undef isnan

#include <future>
#include <chrono>

#include <boost/compute.hpp>

//--------------------------------------------------------------

unsigned w = 1024;
unsigned h = 768;

//--------------------------------------------------------------

template <typename T>
T bounded(T val, T mi, T ma)
{
    return (std::min)((std::max)(val, mi), ma);
}

template <typename T>
T sign(T val)
{
    return val > 0 ? 1 : val < 0 ? -1 : 0;
}

//--------------------------------------------------------------

class reaction
{
public:
    typedef std::vector<float> grid_type;

    reaction(size_t w, size_t h)
        : m_w(w)
        , m_h(h)
        , m_grid(w * h * 2)
    {
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                comp(i, j, 0) = 1;
                comp(i, j, 1) = 0;
            }
        }

        for (int j = h / 2 - 5; j < h / 2 + 5; j++) {
            for (int i = w / 2 - 5; i < w / 2 + 5; i++) {
                comp(i, j, 1) = 1;
            }
        }
    }

    void to_image(ofImage & image) const
    {
        if (!image.isAllocated())
        {
            image.allocate(w, h, OF_IMAGE_GRAYSCALE);
            image.setColor(ofColor::white);
        }

        for (int j = 0; j < m_h; j++) {
            for (int i = 0; i < m_w; i++) {
                image.setColor(i, j, color(comp(i, j, 0), comp(i, j, 1)));
            }
        }
        image.update();
    }

protected:
    float comp(size_t i, size_t j, size_t c) const
    {
        return m_grid[(j * m_w + i) * 2 + c];
    }

    float & comp(size_t i, size_t j, size_t c)
    {
        return m_grid[(j * m_w + i) * 2 + c];
    }

    static ofColor color(float a, float b)
    {
        //float val = bounded((a - b - 0.5f) * 2.0f, -0.5f, 0.5f) + 0.5;
        float val = bounded((a - b) * 2.0f, 0.0f, 1.0f);
        return ofColor(255 * val);
    }

    size_t m_w, m_h;
    grid_type m_grid;
};

//--------------------------------------------------------------

class cpu_reaction
    : public reaction
{
public:
    cpu_reaction(size_t w, size_t h)
        : reaction(w, h)
        , m_grid_other(m_grid)
        , m_current(0)
    {}

    void calc()
    {
        size_t next = (m_current + 1) % 2;
        grid_type const& grid_curr = (m_current == 0 ? m_grid : m_grid_other);
        grid_type & grid_next = (m_current == 0 ? m_grid_other : m_grid);
        m_current = next;

        size_t h1 = m_h / 6;
        size_t h2 = m_h / 3;
        size_t h3 = m_h / 2;
        size_t h4 = 2 * m_h / 3;
        size_t h5 = 5 * m_h / 6;

        auto f1 = std::async(std::launch::async, calculator(grid_curr, grid_next, 1, h1, m_w));
        auto f2 = std::async(std::launch::async, calculator(grid_curr, grid_next, h1, h2, m_w));
        auto f3 = std::async(std::launch::async, calculator(grid_curr, grid_next, h2, h3, m_w));
        auto f4 = std::async(std::launch::async, calculator(grid_curr, grid_next, h3, h4, m_w));
        auto f5 = std::async(std::launch::async, calculator(grid_curr, grid_next, h4, h5, m_w));
        calculator(grid_curr, grid_next, h5, m_h - 1, m_w)();

        f1.get();
        f2.get();
        f3.get();
        f4.get();
        f5.get();
    }

    void next(unsigned n = 1)
    {
        for (unsigned i = 0; i < n; ++i)
            calc();
    }

private:
    class calculator
    {
    public:
        calculator(grid_type const& current, grid_type & next, size_t j_begin, size_t j_end, size_t w)
            : m_current(current)
            , m_next(next)
            , m_j_begin(j_begin)
            , m_j_end(j_end)
            , m_w(w)
        {}

        void operator()()
        {
            float da = 1.0f;
            float db = 0.5f;
            float f = 0.04f;
            float k = 0.0649f;

            for (int j = m_j_begin; j < m_j_end; j++) {
                for (int i = 1; i < m_w - 1; i++) {
                    
                    size_t ai = (j * m_w + i) * 2;
                    size_t bi = ai + 1;

                    //float k = float(i) / m_w * (0.07f - 0.045f) + 0.045f;
                    //float f = float(j) / m_h * (0.1f - 0.01f) + 0.01f;

                    float a = m_current[ai];
                    float b = m_current[bi];
                    float la = laplace(0, m_current, i, j);
                    float lb = laplace(1, m_current, i, j);
                    float abb = a * b * b;
                    float na = a + (da * la - abb + f * (1 - a));
                    float nb = b + (db * lb + abb - (k + f) * b);

                    m_next[ai] = bounded(na, 0.0f, 1.0f);
                    m_next[bi] = bounded(nb, 0.0f, 1.0f);
                }
            }
        }

    private:
        float laplace(size_t comp_index, grid_type const& grid, size_t i, size_t j)
        {
            size_t i_0 = ((j - 1) * m_w + i) * 2;
            size_t i_1 = (j * m_w + i) * 2;
            size_t i_2 = ((j + 1) * m_w + i) * 2;

            return      0.05f * grid[i_0 - 2 + comp_index]
                      + 0.2f * grid[i_0 + comp_index]
                      + 0.05f * grid[i_0 + 2 + comp_index]
                      + 0.2f * grid[i_1 - 2 + comp_index]
                      - 1 * grid[i_1 + comp_index]
                      + 0.2f * grid[i_1 + 2 + comp_index]
                      + 0.05f * grid[i_2 - 2 + comp_index]
                      + 0.2f * grid[i_2 + comp_index]
                      + 0.05f * grid[i_2 + 2 + comp_index];
        }

        grid_type const& m_current;
        grid_type & m_next;
        size_t m_j_begin;
        size_t m_j_end;
        size_t m_w;
    };

    grid_type m_grid_other;
    size_t m_current;
};

//--------------------------------------------------------------

namespace bc = boost::compute;

class gpu_reaction
    : public reaction
{
public:
    typedef std::vector<float> grid_type;

    gpu_reaction(size_t w, size_t h)
        : reaction(w, h)
        , m_device(bc::system::default_device())
        , m_context(m_device)
        , m_queue(m_context, m_device)
        , m_device_vectors{ bc::vector<float>(w * h * 2, m_context), bc::vector<float>(w * h * 2, m_context) }
        , m_current(0)
        , m_program(m_context)
        , m_kernel(m_program.program, "my_program")
    {
        bc::copy(m_grid.begin(), m_grid.end(), m_device_vectors[0].begin(), m_queue);
        bc::copy(m_device_vectors[0].begin(), m_device_vectors[0].end(), m_device_vectors[1].begin(), m_queue);
    }

    void calc()
    {
        size_t next = (m_current + 1) % 2;
        bc::vector<float> const& current_vector = m_device_vectors[m_current];
        bc::vector<float> const& next_vector = m_device_vectors[next];
        m_current = next;

        m_kernel.set_arg(0, current_vector.get_buffer());
        m_kernel.set_arg(1, next_vector.get_buffer());
        m_kernel.set_arg(2, (bc::uint_)m_w);
        m_kernel.set_arg(3, (bc::uint_)m_h);

        size_t origin[2] = { 1, 1 };
        size_t region[2] = { m_w - 2, m_h - 2 };

        m_queue.enqueue_nd_range_kernel(m_kernel, 2, origin, region, 0);
        m_queue.finish();
    }

    void next(unsigned n = 1)
    {
        for (unsigned i = 0 ; i < n ; ++i)
            calc();

        bc::vector<float> const& current_vector = m_device_vectors[m_current];
        bc::copy(current_vector.begin(), current_vector.end(), m_grid.begin(), m_queue);
    }

private:
    struct program_holder
    {
        program_holder(bc::context const& context)
            : program(bc::program::create_with_source(source(), context))
        {
            program.build();
        }

        static const char * source()
        {
            return BOOST_COMPUTE_STRINGIZE_SOURCE(
                __kernel void my_program(__global __read_only float* curr,
                                         __global __write_only float* next,
                                         uint w,
                                         uint h)
                {
                    uint i = get_global_id(0);
                    uint j = get_global_id(1);
                
                    float da = 1.0f;
                    float db = 0.5f;
                    //float f = 0.04f;
                    //float k = 0.0649f;
                    //float f = 0.0545f;
                    //float k = 0.062f;
                    //float f = 0.055f;
                    //float k = 0.062f;
                    float f = (float)j / h * (0.07f - 0.03f) + 0.01f;
                    float k = (float)i / w * (0.07f - 0.045f) + 0.045f;

                    int ia = (j * w + i) * 2;
                    int ib = ia + 1;

                    int ia_0 = ((j - 1) * w + i) * 2;
                    int ib_0 = ia_0 + 1;
                    int ia_2 = ((j + 1) * w + i) * 2;
                    int ib_2 = ia_2 + 1;

                    float a = curr[ia];
                    float b = curr[ib];
                    
                    float la = 0.05f * curr[ia_0 - 2]
                        + 0.2f * curr[ia_0]
                        + 0.05f * curr[ia_0 + 2]
                        + 0.2f * curr[ia - 2]
                        - 1 * a
                        + 0.2f * curr[ia + 2]
                        + 0.05f * curr[ia_2 - 2]
                        + 0.2f * curr[ia_2]
                        + 0.05f * curr[ia_2 + 2];

                    float lb = 0.05f * curr[ib_0 - 2]
                        + 0.2f * curr[ib_0]
                        + 0.05f * curr[ib_0 + 2]
                        + 0.2f * curr[ib - 2]
                        - 1 * b
                        + 0.2f * curr[ib + 2]
                        + 0.05f * curr[ib_2 - 2]
                        + 0.2f * curr[ib_2]
                        + 0.05f * curr[ib_2 + 2];

                    float abb = a * b * b;
                    float na = a + (da * la - abb + f * (1 - a));
                    float nb = b + (db * lb + abb - (k + f) * b);

                    if (na < 0.0f) na = 0.0f;
                    else if (na > 1.0f) na = 1.0f;
                    if (nb < 0.0f) nb = 0.0f;
                    else if (nb > 1.0f) nb = 1.0f;

                    next[ia] = na;
                    next[ib] = nb;
                }
            );
        }

        bc::program program;
    };

    bc::device m_device;
    bc::context m_context;
    bc::command_queue m_queue;
    
    bc::vector<float> m_device_vectors[2];
    size_t m_current;

    program_holder m_program;
    bc::kernel m_kernel;
};

//--------------------------------------------------------------

ofImage image1, image2;
cpu_reaction cpu_react(w, h);
gpu_reaction gpu_react(w, h);

//--------------------------------------------------------------
void ofApp::setup() {
    ofBackground(255);

    ofSetFrameRate(0);
    ofSetVerticalSync(false);
}

//--------------------------------------------------------------
void ofApp::update() {
    typedef std::chrono::time_point<std::chrono::system_clock> time_point;
    
    /*{
        time_point start = std::chrono::system_clock::now();

        cpu_react.next(2);

        time_point end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << " ";
    }*/

    {
        //time_point start = std::chrono::system_clock::now();

        gpu_react.next(64);
        
        //time_point end = std::chrono::system_clock::now();
        //std::chrono::duration<double> elapsed_seconds = end - start;
        //std::cout << elapsed_seconds.count() << " ";
    }

    //cpu_react.to_image(image1);
    gpu_react.to_image(image2);
}

//--------------------------------------------------------------
void ofApp::draw() {
    int w = ofGetViewportWidth();
    int h = ofGetViewportHeight();
    //image1.draw(0, 0, w/2, h);
    //image2.draw(w/2, 0, w / 2, h);
    image2.draw(0, 0, w, h);

    std::cout << ofGetFrameRate() << "FPS" << std::endl;
}

//--------------------------------------------------------------

void ofApp::keyPressed(int key) {
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
