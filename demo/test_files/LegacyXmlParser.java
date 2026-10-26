package org.example.xml;

import java.io.InputStream;

/**
 * Parses configuration files written in the old XML format.
 * <p>
 * The parser understands the following elements:
 * <ul>
 *   <li>{@code <server>} with a host and a port</li>
 *   <li>{@code <user>} with a name and a role</li>
 * </ul>
 * It is kept only for backwards compatibility, because some customers still deploy XML configuration files.
 * New code should read YAML files with {@link YamlConfigReader}.
 *
 * @author Jane Doe
 * @author John Smith
 * @deprecated since 4.0, use {@link YamlConfigReader} instead; this class will be removed in 5.0.
 */
@Deprecated(since = "4.0", forRemoval = true)
public class LegacyXmlParser {

    /* Not a Javadoc comment: ignored by the demo. */
    public Object parse(InputStream in) {
        return null;
    }
}
